let model;
let scalerMean = null;
let scalerStd = null;

// ----------------------
// Load scaler.pkl
// ----------------------
async function loadScaler() {
    const response = await fetch("scaler/nutrition_scaler.json");
    const data = await response.json();

    scalerMean = data.mean;
    scalerStd  = data.scale;

    console.log("Scaler loaded:", scalerMean, scalerStd);
}

// ----------------------
// Inverse StandardScaler
// ----------------------
function inverseTransform(pred) {
    return pred.map((v, i) => v * scalerStd[i] + scalerMean[i]);
}

// ----------------------
// Load TF.js Model
// ----------------------
async function loadModel() {
    model = await tf.loadGraphModel("model/model.json");
    console.log("Model loaded.");
}

// ----------------------
// Predict
// ----------------------
async function predictFromImage(imgElement) {
    let tensor = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims(0);

    const prediction = model.predict(tensor);
    const output = await prediction.data();
    const result = inverseTransform(Array.from(output));

    document.getElementById("result").innerHTML = `
        <h3>Estimated Nutrition</h3>
        <p><b>Calories:</b> ${result[0].toFixed(1)}</p>
        <p><b>Protein:</b> ${result[1].toFixed(1)} g</p>
        <p><b>Fat:</b> ${result[2].toFixed(1)} g</p>
        <p><b>Carbs:</b> ${result[3].toFixed(1)} g</p>
    `;
}

// ----------------------
// Init + file input
// ----------------------
window.onload = async () => {
    await loadScaler();
    await loadModel();

    document.getElementById("fileInput").addEventListener("change", e => {
        let file = e.target.files[0];
        let img = document.getElementById("preview");
        img.src = URL.createObjectURL(file);
        img.style.display = "block";

        img.onload = () => predictFromImage(img);
    });
};
