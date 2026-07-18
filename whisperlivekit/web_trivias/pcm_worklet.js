class PCMForwarder extends AudioWorkletProcessor {
  constructor() {
    super();
    this._threshold = 0;     // 0 = gate uitgeschakeld
    this._holdFrames = 113;  // ~300ms bij 48kHz / 128 samples per frame
    this._holdCounter = 0;
    this._gateOpen = false;

    this.port.onmessage = (e) => {
      if (e.data.threshold !== undefined) {
        this._threshold = e.data.threshold;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || !input[0].length) return true;

    const channelData = input[0];

    // RMS altijd berekenen (ook bij threshold=0), nodig voor cross-kanaal arbitrage
    let sum = 0;
    for (let i = 0; i < channelData.length; i++) {
      sum += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sum / channelData.length);

    if (this._threshold > 0) {
      if (rms >= this._threshold) {
        // Stem gedetecteerd: gate openen en hold-teller resetten
        this._gateOpen = true;
        this._holdCounter = this._holdFrames;
      } else if (this._holdCounter > 0) {
        // Nog binnen hold-periode: gate blijft open
        this._holdCounter--;
      } else {
        // Onder drempel en hold voorbij: gate sluiten
        this._gateOpen = false;
      }
    } else {
      this._gateOpen = true;
    }

    // Altijd versturen: audio is leidend (WAV-opname), gateOpen bepaalt alleen of
    // dit fragment naar ASR mag — nooit of het wordt opgenomen.
    const copy = new Float32Array(channelData.length);
    copy.set(channelData);
    this.port.postMessage({ buffer: copy, rms, gateOpen: this._gateOpen }, [copy.buffer]);
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMForwarder);
