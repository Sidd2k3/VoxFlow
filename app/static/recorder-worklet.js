// Runs in the browser's audio thread: passes each block of microphone samples to app.js.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("recorder", RecorderProcessor);
