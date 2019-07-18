function createLatentVector(size) {
    var latent_vector = new Float32Array(size);

    for (i = 0; i < latent_vector.length; ++i) {
        latent_vector[i] = Math.random()// * 2 - 1
    }

    return new onnx.Tensor(latent_vector, 'float32', [1, size])
}


function getInputs() {
    var slider = document.getElementById("myRange");
    latent_vector.set(Number(slider.value), 0, 12)

    return latent_vector
}


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

var latent_vector = createLatentVector(100)



canvas = document.getElementById("myCanvas");
buffer = new Uint8ClampedArray(32 * 32 * 4).fill(255);
ctx = canvas.getContext("2d");
idata = ctx.createImageData(32, 32);

const session = new onnx.InferenceSession()
// load the ONNX model file

modelLoaded = session.loadModel("http://localhost:5000/generator.onnx");
modelLoaded.then(evaluateModel())

