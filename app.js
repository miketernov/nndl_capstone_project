console.log("Loading TFJS model...");

// ======== LOAD MODEL ========
let model = null;
async function loadModel() {
    model = await tf.loadLayersModel("tfjs_model/model.json");
    console.log("Model loaded:", model);
}

loadModel();

// ======== LOAD SCALER ========
let scaler = null;

async function loadScaler() {
    try {
        const resp = await fetch("nutrition_scaler.json");
        scaler = await resp.json();
        console.log("Scaler loaded:", scaler);
    } catch (e) {
        console.error("Scaler loading failed:", e);
    }
}

loadScaler();

// ======== IMAGE PREPROCESSING ========
function preprocessImage(img) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0);

        return tensor.expandDims(0); // [1,224,224,3]
    });
}

// ======== APPLY SCALER ========
function descaleOutput(pred) {
    if (!scaler) return pred; // fallback

    const mean = tf.tensor(scaler.mean);
    const scale = tf.tensor(scaler.scale);

    return pred.mul(scale).add(mean);
}

// ======== MAIN PREDICTION ========
async function predictImage(imgElement) {
    if (!model) {
        alert("Model not loaded yet!");
        return;
    }
    const input = preprocessImage(imgElement);

    const rawPred = model.predict(input);
    const unscaled = descaleOutput(rawPred);

    const values = await unscaled.data();

    rawPred.dispose();
    unscaled.dispose();
    input.dispose();

    return values;
}

// ======== UI HANDLER ========
document.getElementById("fileInput").addEventListener("change", async function (event) {
    const file = event.target.files[0];
    if (!file) return;

    let img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        const result = await predictImage(img);
        console.log("Prediction:", result);

        document.getElementById("output").innerHTML = `
            <h3>Nutrition estimation</h3>
            <p>Calories: ${result[0].toFixed(2)}</p>
            <p>Proteins: ${result[1].toFixed(2)}</p>
            <p>Fats: ${result[2].toFixed(2)}</p>
            <p>Carbs: ${result[3].toFixed(2)}</p>
        `;
    };
});
