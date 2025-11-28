let model = null;
let scalerMean = null;
let scalerStd = null;

// ----------------------
// Load model + scaler
// ----------------------
async function loadResources() {
    console.log("Loading TFJS model...");
    model = await tf.loadLayersModel("model/model.json");
    console.log("Model loaded.");

    console.log("Loading scaler...");
    const response = await fetch("scaler/nutrition_scaler.json");
    const data = await response.json();
    scalerMean = data.mean;
    scalerStd = data.scale;
    console.log("Scaler loaded:", scalerMean, scalerStd);
}

loadResources();


// ----------------------
// Image preprocessing
// ----------------------
function preprocessImage(imgElement) {
    let tensor = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims(0); // [1, 224, 224, 3]

    return tensor;
}


// ----------------------
// Inverse scaling
// ----------------------
function inverseScale(arr) {
    let out = [];
    for (let i = 0; i < arr.length; i++) {
        out.push(arr[i] * scalerStd[i] + scalerMean[i]);
    }
    return out;
}


// ----------------------
// Prediction pipeline
// ----------------------
async function predictNutrition() {
    const img = document.getElementById("preview");

    if (!model || !scalerMean) {
        alert("Model or scaler not loaded yet. Please wait 1–2 seconds.");
        return;
    }

    const input = preprocessImage(img);
    const pred = model.predict(input);
    const values = await pred.data();

    const real = inverseScale(values);

    displayResult(real);
}


// ----------------------
// Update UI
// ----------------------
function displayResult(v) {
    const box = document.getElementById("result");
    box.innerHTML = `
        <h3>Estimated Nutrition</h3>
        <p><b>Calories:</b> ${v[0].toFixed(1)}</p>
        <p><b>Protein:</b> ${v[1].toFixed(1)} g</p>
        <p><b>Fat:</b> ${v[2].toFixed(1)} g</p>
        <p><b>Carbs:</b> ${v[3].toFixed(1)} g</p>
    `;
    box.style.display = "block";
}


// ----------------------
// File input → preview
// ----------------------
document.getElementById("fileInput").addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    img.onload = () => {
        predictNutrition();
    };
});
