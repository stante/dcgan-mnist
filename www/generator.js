// create a session
function getInputs() {
    var slider = document.getElementById("myRange");
    var latent_vector = new Float32Array(100).fill(1)
    latent_vector[3] = slider.value
    latent_vector[1] = slider.value
    return new onnx.Tensor(latent_vector, 'float32', [1, 100])
}

function setBuffer(buffer, tensor) {
    for (y = 0; y < 32; ++y) {
        for (x = 0; x < 32; ++x) {
            index = y * 32 + x
            value = (tensor.get(0, 0, y, x) + 1) / 2 * 255
            // console.log(tensor.get(0, 0, x, y));
            buffer[index * 4] = value
            buffer[index * 4 + 1] = value
            buffer[index * 4 + 2] = value
            buffer[index * 4 + 3] = 255
        }
    }
}

buffer = new Uint8ClampedArray(32 * 32 * 4).fill(255);
c = document.getElementById("myCanvas");
ctx = c.getContext("2d");
ctx.scale(10, 10)
idata = ctx.createImageData(32, 32);

const session = new onnx.InferenceSession()
// load the ONNX model file

modelLoaded = session.loadModel("https://rdwr.org/generator.onnx");
modelLoaded.then(evaluateModel())

function evaluateModel() {
    modelLoaded.then(() => {
    // generate model input
    const inferenceInputs = getInputs();
    //console.log(`model input tensor: ${inferenceInputs.data}.`);
    // execute the model
    session.run([inferenceInputs]).then(output => {
        // consume the output
        const outputTensor = output.values().next().value;
        //console.log(`Before: ${buffer}`)
        setBuffer(buffer, outputTensor)
        //console.log(`After: ${buffer}`)
        // console.log(`model output: ${outputTensor.data.length}`)
        //console.log(`model output tensor: ${outputTensor.data}.`);
        idata.data.set(buffer);

        // update canvas with new data
        ctx.putImageData(idata, 0, 0);
    });
    });
}