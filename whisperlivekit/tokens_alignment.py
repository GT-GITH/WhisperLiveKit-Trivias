from time import time
from typing import Any, List, Optional, Tuple, Union

from whisperlivekit.timed_objects import (ASRToken, Segment, PuncSegment, Silence,
                                          SilentSegment, SpeakerSegment,
                                          TimedText)
import logging

logger = logging.getLogger("whisperlivekit.tokens_alignment")
# Geen basicConfig en geen forced setLevel hier.
# De applicatie (server) bepaalt het loglevel.

_PRUNE_MAX_DURATION_S: float = 600.0  # maximaal 10 minuten tokens in memory
_PRUNE_TRIGGER_COUNT: int = 13_000    # alleen uitvoeren als de lijst deze grens overschrijdt

class TokensAlignment:

    def __init__(self, state: Any, args: Any, sep: Optional[str]) -> None:
        self.state = state
        self.diarization = args.diarization
        self._tokens_index: int = 0
        self._diarization_index: int = 0
        self._translation_index: int = 0

        self.all_tokens: List[ASRToken] = []
        self.all_diarization_segments: List[SpeakerSegment] = []
        self.all_translation_segments: List[Any] = []

        self.new_tokens: List[ASRToken] = []
        self.new_diarization: List[SpeakerSegment] = []
        self.new_translation: List[Any] = []
        self.new_translation_buffer: Union[TimedText, str] = TimedText()
        self.new_tokens_buffer: List[Any] = []
        self.sep: str = sep if sep is not None else ' '
        self.beg_loop: Optional[float] = None

        self.validated_segments: List[Segment] = []
        self.current_line_tokens: List[ASRToken] = []
        self.diarization_buffer: List[ASRToken] = []

        self.last_punctuation = None
        self.last_uncompleted_punc_segment: PuncSegment = None
        self.unvalidated_tokens: PuncSegment = []

        # Debug throttling (voorkomt log-spam)
        self._dbg_last_sig: Optional[str] = None
        self._dbg_last_live: Optional[str] = None
        self._dbg_last_log_t: float = 0.0
        self._dbg_min_interval_s: float = 0.75  # max ~1x per 0.75s
        
        self.segment_overrides = {}  # id -> dict(state,text_batch,text_final,start_ms,end_ms)
        # A1: canonical batch groups (server-side)
        # We store these as segments and suppress any normal segments inside the refined window.
        self.batch_groups: List[Segment] = []
        self.suppressed_ranges_ms: List[Tuple[int, int]] = []  # list of (start_ms,end_ms)

    def set_segment_override(
        self,
        seg_id: str,
        *,
        state: str = "FINAL",
        text_batch: Optional[str] = None,
        text_final: Optional[str] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None
    ) -> None:
        self.segment_overrides[seg_id] = {
            "state": state,
            "text_batch": text_batch,
            "text_final": text_final,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

    def apply_batch_group(
        self,
        *,
        window_start_ms: int,
        window_end_ms: int,
        text_final: str,
        text_batch: Optional[str] = None,
        speaker: int = -1,
    ) -> str:
        """
        A1: Maak één canonical BatchGroup-segment voor dit window, en suppress alle bestaande
        (validated) segmenten die overlappen met dit window.
        Returns: group_id
        """
        if window_end_ms <= window_start_ms:
            return ""

        group_id = f"bg_{int(window_start_ms)}_{int(window_end_ms)}_{int(speaker)}"

        # 1) Suppress ranges (keep list non-overlapping-ish is optional; simple append is ok)
        self.suppressed_ranges_ms.append((int(window_start_ms), int(window_end_ms)))

        # 2) Drop validated segments that overlap with this refined window
        kept: List[Segment] = []
        for s in self.validated_segments:
            try:
                s_start_ms = int(round(float(getattr(s, "start", 0.0) or 0.0) * 1000.0))
                s_end_s = getattr(s, "end", None)
                s_end_ms = int(round(float(s_end_s) * 1000.0)) if s_end_s is not None else s_start_ms
            except Exception:
                kept.append(s)
                continue

            # overlap?
            if s_end_ms >= window_start_ms and s_start_ms < window_end_ms:
                # drop it (including silence segments) → BatchGroup becomes canonical truth
                continue

            kept.append(s)

       # als uitgeschakeld: Dan blijft live zichtbaar naast batch canonical groups. Voor debug is dit perfect.
        self.validated_segments = kept

        # 3) Remove older batch groups that overlap the same window (defensive)
        new_groups: List[Segment] = []
        for g in self.batch_groups:
            try:
                g_start_ms = int(round(float(getattr(g, "start", 0.0) or 0.0) * 1000.0))
                g_end_s = getattr(g, "end", None)
                g_end_ms = int(round(float(g_end_s) * 1000.0)) if g_end_s is not None else g_start_ms
            except Exception:
                new_groups.append(g)
                continue

            if g_end_ms > window_start_ms and g_start_ms < window_end_ms:
                # drop overlapping older group
                continue
            new_groups.append(g)

        self.batch_groups = new_groups

        # 4) Create canonical group segment
        grp = Segment()
        grp.id = group_id
        grp.start = window_start_ms / 1000.0
        grp.end = window_end_ms / 1000.0
        grp.speaker = speaker
        grp.state = "FINAL"
        grp.text = (text_final or "").strip()
        grp.text_batch = (text_batch or "").strip() if text_batch else None
        grp.text_live = None

        self.batch_groups.append(grp)

        return group_id

    def update(self) -> None:
        """Drain state buffers into the running alignment context."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
        self.new_translation_buffer = self.state.new_translation_buffer

        if len(self.all_tokens) > _PRUNE_TRIGGER_COUNT:
            self._prune_buffers()

    def _prune_buffers(self) -> None:
        """Verwijder tokens en segmenten ouder dan _PRUNE_MAX_DURATION_S.

        Tokens die nog niet zijn verwerkt tot een validated_segment worden
        beschermd via current_line_tokens en unvalidated_tokens.
        """
        if not self.all_tokens:
            return

        newest_end = 0.0
        for t in self.all_tokens:
            e = getattr(t, 'end', None)
            if e is not None and e > newest_end:
                newest_end = e
        horizon = newest_end - _PRUNE_MAX_DURATION_S
        if horizon <= 0.0:
            return

        # Bescherm tokens die nog in de live tail zitten (nog niet gevalideerd)
        protected = float('inf')
        for t in self.current_line_tokens:
            s = getattr(t, 'start', None)
            if s is not None and s < protected:
                protected = s
        for t in self.unvalidated_tokens:
            s = getattr(t, 'start', None)
            if s is not None and s < protected:
                protected = s
        cutoff = min(horizon, protected)
        if cutoff <= 0.0:
            return

        n_before = len(self.all_tokens)
        self.all_tokens = [
            t for t in self.all_tokens
            if (getattr(t, 'end', None) is None or t.end >= cutoff)
        ]
        pruned = n_before - len(self.all_tokens)
        if pruned:
            logger.debug(
                "[PRUNE] all_tokens -%d → %d (cutoff=%.1fs)",
                pruned, len(self.all_tokens), cutoff,
            )

        if self.all_diarization_segments:
            self.all_diarization_segments = [
                s for s in self.all_diarization_segments
                if (getattr(s, 'end', None) is None or s.end >= cutoff)
            ]

        if self.all_translation_segments:
            self.all_translation_segments = [
                s for s in self.all_translation_segments
                if (getattr(s, 'end', None) is None or s.end >= cutoff)
            ]

    def add_translation(self, segment: Segment) -> None:
        """Append translated text segments that overlap with a segment."""
        if segment.translation is None:
            segment.translation = ''
        for ts in self.all_translation_segments:
            if ts.is_within(segment):
                segment.translation += ts.text + (self.sep if ts.text else '')
            elif segment.translation:
                break


    def compute_punctuations_segments(self, tokens: Optional[List[ASRToken]] = None) -> List[PuncSegment]:
        """Group tokens into segments split by punctuation and explicit silence."""
        segments = []
        segment_start_idx = 0
        for i, token in enumerate(self.all_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i+1],
                    )
                    segments.append(segment)
                    segment_start_idx = i+1

        final_segment = PuncSegment.from_tokens(
            tokens=self.all_tokens[segment_start_idx:],
        )
        if final_segment:
            segments.append(final_segment)
        return segments

    def compute_new_punctuations_segments(self) -> List[PuncSegment]:
        new_punc_segments = []
        segment_start_idx = 0
        self.unvalidated_tokens += self.new_tokens
        for i, token in enumerate(self.unvalidated_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    new_punc_segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                new_punc_segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i+1],
                    )
                    new_punc_segments.append(segment)
                    segment_start_idx = i+1

        self.unvalidated_tokens = self.unvalidated_tokens[segment_start_idx:]
        return new_punc_segments


    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
        """Merge consecutive diarization slices that share the same speaker."""
        if not self.all_diarization_segments:
            return []
        merged = [self.all_diarization_segments[0]]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(segment)
        return merged


    @staticmethod
    def intersection_duration(seg1: TimedText, seg2: TimedText) -> float:
        """Return the overlap duration between two timed segments."""
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def get_lines_diarization(self) -> Tuple[List[Segment], str]:
        """Build segments when diarization is enabled and track overflow buffer."""
        diarization_buffer = ''
        punctuation_segments = self.compute_punctuations_segments()
        diarization_segments = self.concatenate_diar_segments()
        for punctuation_segment in punctuation_segments:
            if not punctuation_segment.is_silence():
                if diarization_segments and punctuation_segment.start >= diarization_segments[-1].end:
                    diarization_buffer += punctuation_segment.text
                else:
                    # NEW: determine speaker by majority of token speakers (defensive)
                    toks = getattr(punctuation_segment, "tokens", None) or []
                    speaker_counts = {}

                    for tok in toks:
                        sp = getattr(tok, "speaker", None)
                        if sp is not None and sp >= 0:
                            speaker_counts[sp] = speaker_counts.get(sp, 0) + 1

                    if speaker_counts:
                        # choose speaker with most tokens
                        punctuation_segment.speaker = max(
                            speaker_counts.items(), key=lambda x: x[1]
                        )[0] + 1
                    else:
                        # fallback: overlap-based speaker assignment (original logic)
                        max_overlap = 0.0
                        max_overlap_speaker = 1
                        for diarization_segment in diarization_segments:
                            intersec = self.intersection_duration(punctuation_segment, diarization_segment)
                            if intersec > max_overlap:
                                max_overlap = intersec
                                max_overlap_speaker = diarization_segment.speaker + 1
                        punctuation_segment.speaker = max_overlap_speaker
        
        segments = []
        if punctuation_segments:
            segments = [punctuation_segments[0]]
            for segment in punctuation_segments[1:]:
                if segment.speaker == segments[-1].speaker:
                    if segments[-1].text:
                        segments[-1].text += segment.text
                    segments[-1].end = segment.end
                else:
                    segments.append(segment)

        return segments, diarization_buffer


    def get_lines(
            self, 
            diarization: bool = False,
            translation: bool = False,
            current_silence: Optional[Silence] = None
        ) -> Tuple[List[Segment], str, Union[str, TimedText]]:
        """Return the formatted segments plus buffers, optionally with diarization/translation."""
        if diarization:
            segments, diarization_buffer = self.get_lines_diarization()
        else:
            diarization_buffer = ''
            for token in self.new_tokens:
                if token.is_silence():
                    if self.current_line_tokens:
                        self.validated_segments.append(Segment().from_tokens(self.current_line_tokens))
                        self.current_line_tokens = []
                    
                    end_silence = token.end if token.has_ended else time() - self.beg_loop
                    if self.validated_segments and self.validated_segments[-1].is_silence():
                        self.validated_segments[-1].end = end_silence
                    else:
                        self.validated_segments.append(SilentSegment(
                            start=token.start,
                            end=end_silence
                        ))
                else:
                    self.current_line_tokens.append(token)

            # tokens_alignment.py (in get_lines, non-diarization pad)
            # Start with validated FINAL segments + canonical batch groups
            segments = list(self.validated_segments) + list(self.batch_groups)

            logger.debug(
                f"[UI MIX] validated={len(self.validated_segments)} "
                f"batch_groups={len(self.batch_groups)} "
                f"current_live_tokens={len(self.current_line_tokens)}"
            )
            if self.current_line_tokens:
                logger.debug(
                    f"[UI LIVE TEXT] tail='{''.join([t.text for t in self.current_line_tokens[-5:]])}'"
                )
            # validated segments are FINAL
            for s in segments:
                # FORCE deterministic ID for ALL segments
                if s.id is None:
                    s.id = f"seg_{int(round((s.start or 0) * 1000))}_{s.speaker}"

                if not s.is_silence():
                    s.state = "FINAL"
                    if s.text_live is None:
                        s.text_live = s.text

                if logger.isEnabledFor(logging.DEBUG):
                    # Log alleen als de "signature" wijzigt én niet te vaak
                    sig = f"{s.id}:{s.state}:{s.start:.2f}->{(s.end or 0):.2f}"
                    now = time()
                    if sig != self._dbg_last_sig and (now - self._dbg_last_log_t) >= self._dbg_min_interval_s:
                        logger.debug(f"SEG {s.id} {s.state} {s.start}->{s.end}")
                        self._dbg_last_sig = sig
                        self._dbg_last_log_t = now


            if self.current_line_tokens:
                live_seg = Segment().from_tokens(self.current_line_tokens)
                if live_seg and not live_seg.is_silence():
                    live_seg.state = "LIVE"
                    # live_seg.text is the provisional display, text_live mirrors it
                    live_seg.text_live = live_seg.text
                    if logger.isEnabledFor(logging.DEBUG):
                        live_sig = f"{live_seg.id}:{live_seg.start:.2f}->{(live_seg.end or 0):.2f}"
                        now = time()
                        if live_sig != self._dbg_last_live and (now - self._dbg_last_log_t) >= self._dbg_min_interval_s:
                            logger.debug(f"LIVE {live_seg.id} {live_seg.start}->{live_seg.end}")
                            self._dbg_last_live = live_sig
                            self._dbg_last_log_t = now

                # Guard: voorkom duplicate live segment met dezelfde ID als laatst gevalideerde segment
                if live_seg:
                    if segments:
                        last = segments[-1]
                        if hasattr(last, "id") and hasattr(live_seg, "id") and last.id == live_seg.id:
                            # Skip duplicate
                            pass
                        else:
                            segments.append(live_seg)
                    else:
                        segments.append(live_seg)


        if current_silence:
            end_silence = current_silence.end if current_silence.has_ended else time() - self.beg_loop
            if segments and segments[-1].is_silence():
                segments[-1] = SilentSegment(start=segments[-1].start, end=end_silence)
            else:
                segments.append(SilentSegment(
                    start=current_silence.start,
                    end=end_silence
                ))
        if translation:
            [self.add_translation(segment) for segment in segments if not segment.is_silence()]

        # Apply batch overrides (persistently)
        for seg in segments:
            ov = self.segment_overrides.get(seg.id)
            if not ov:
                continue

            if ov.get("state"):
                seg.state = ov["state"]

            if ov.get("text_batch") is not None:
                seg.text_batch = ov["text_batch"]

            if ov.get("text_final") is not None:
                seg.text = ov["text_final"]      # 'text' is what UI shows as final
                seg.text_live = None             # optional: hide live after final

            # Optional: align boundaries to batch window
            if ov.get("start_ms") is not None:
                seg.start = ov["start_ms"] / 1000.0
            if ov.get("end_ms") is not None:
                seg.end = ov["end_ms"] / 1000.0

        # --- Prune segments that overlap canonical batch windows ---
        # Canonical windows = batch_groups or suppressed_ranges_ms

        def _seg_ms(seg: Segment) -> Tuple[int, int]:
            start_ms = int(round(float(getattr(seg, "start", 0.0) or 0.0) * 1000.0))
            end_s = getattr(seg, "end", None)
            end_ms = int(round(float(end_s) * 1000.0)) if end_s is not None else start_ms
            return start_ms, end_ms

        canonical_ranges: List[Tuple[int, int]] = []
        canonical_ids = set()

        canonical_ranges.extend(list(self.suppressed_ranges_ms))

        for g in self.batch_groups:
            try:
                canonical_ranges.append(_seg_ms(g))
                if getattr(g, "id", None):
                    canonical_ids.add(g.id)
            except Exception:
                pass

        if canonical_ranges:
            pruned: List[Segment] = []

            for s in segments:

                seg_id = getattr(s, "id", None)

                # Batchgroup zelf altijd behouden
                if seg_id in canonical_ids or (isinstance(seg_id, str) and seg_id.startswith("bg_")):
                    pruned.append(s)
                    continue

                s_start, s_end = _seg_ms(s)

                overlaps = False
                for c_s, c_e in canonical_ranges:
                    if (s_end >= c_s) and (s_start < c_e):
                        overlaps = True
                        break

                if overlaps:
                    continue

                pruned.append(s)

            segments = pruned
            
        # --- Ensure deterministic chronological ordering (CRITICAL) ---
        def _seg_sort_key(s: Segment):
            start_ms = int(round(float(getattr(s, "start", 0.0) or 0.0) * 1000.0))
            end_s = getattr(s, "end", None)
            end_ms = int(round(float(end_s) * 1000.0)) if end_s is not None else start_ms

            # Order: time asc, FINAL before LIVE, shorter first if same start
            state = getattr(s, "state", "") or ""
            state_rank = 0 if state == "FINAL" else 1  # FINAL first

            # Silence last if same start (optional but keeps text nicer)
            silence_rank = 1 if hasattr(s, "is_silence") and s.is_silence() else 0

            return (start_ms, state_rank, silence_rank, end_ms)

        segments.sort(key=_seg_sort_key)


        return segments, diarization_buffer, self.new_translation_buffer.text
 