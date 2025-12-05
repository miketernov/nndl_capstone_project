// -------------------------------------------------------------
// GLOBAL VARIABLES
// -------------------------------------------------------------
let model = null;

const NUTRITION_MEAN = [257.7, 214.42, 12.97, 19.28, 18.26];
const NUTRITION_STD = [211.42, 153.17, 13.72, 16.17, 20.14];

let dailyStats = { calories: 0, protein: 0, fat: 0, carbs: 0 };
let dailyPlan = null;
let mealHistory = [];

// ------------------------ THEME SWITCH ------------------------
function toggleTheme() {
  document.documentElement.classList.toggle("light");
}

// -------------------------------------------------------------
// LOAD MODEL
// -------------------------------------------------------------
function getModelUrl() {
  const base = window.location.href.replace(/index\.html?$/, "");
  return new URL("model.json", base).toString();
}

window.addEventListener("load", async () => {
  const s = document.getElementById("status");

  try {
    s.textContent = "Loading model...";
    model = await tf.loadGraphModel(getModelUrl());
    s.textContent = "Model loaded. Upload a meal.";
  } catch (e) {
    console.error("Model load error:", e);
    s.textContent = "Failed to load model.";
  }

  document.getElementById("file-input").addEventListener("change", handleFileChange);

  initCharts();
});

// -------------------------------------------------------------
// USER PLAN CALCULATION
// -------------------------------------------------------------
function calculatePlan() {
  const age = +document.getElementById("age").value;
  const weight = +document.getElementById("weight").value;
  const height = +document.getElementById("height").value;
  const activity = +document.getElementById("activity").value;
  const goal = document.getElementById("goal").value;

  if (!age || !weight || !height) {
    alert("Please fill all fields.");
    return;
  }

  let bmr = 10 * weight + 6.25 * height - 5 * age + 5;
  let tdee = bmr * activity;

  if (goal === "loss") tdee *= 0.82;
  if (goal === "gain") tdee *= 1.15;

  const protein = weight * 1.8;
  const fat = weight * 0.9;
  const carbs = (tdee - protein * 4 - fat * 9) / 4;

  dailyPlan = { tdee, protein, fat, carbs };

  document.getElementById("recommendations").innerHTML = `
    <p><b>Daily calories:</b> ${Math.round(tdee)} kcal</p>
    <p><b>Protein:</b> ${Math.round(protein)} g</p>
    <p><b>Fat:</b> ${Math.round(fat)} g</p>
    <p><b>Carbs:</b> ${Math.round(carbs)} g</p>
  `;

  renderDailyProgress();
  updateTips();
}

// -------------------------------------------------------------
// IMAGE HANDLING
// -------------------------------------------------------------
function handleFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (!model) return alert("Model not loaded.");

  const reader = new FileReader();
  reader.onload = () => {
    const img = document.getElementById("preview");

    img.onload = async () => {
      document.getElementById("status").textContent = "Predicting...";
      const pred = await runInferenceOnImage(img);

      updateResults(pred);
      updateDailyStats(pred);
      updateCharts(pred);
      renderHistory();
      updateTips();

      document.getElementById("status").textContent = "Done.";
    };

    img.src = reader.result;
  };

  reader.readAsDataURL(file);
}

// -------------------------------------------------------------
// MODEL INFERENCE
// -------------------------------------------------------------
function runInferenceOnImage(img) {
  return tf.tidy(() => {
    let x = tf.browser.fromPixels(img).toFloat();
    x = tf.image.resizeBilinear(x, [224, 224], true);
    x = x.expandDims(0);

    const y = model.execute(x, "Identity:0");
    const vals = y.dataSync();

    const den = vals.map((v, i) => v * NUTRITION_STD[i] + NUTRITION_MEAN[i]);

    return {
      calories: round1(den[0]),
      mass: round1(den[1]),
      fat: round1(den[2]),
      carbs: round1(den[3]),
      protein: round1(den[4])
    };
  });
}

function round1(x) { return Math.round(x * 10) / 10; }

// -------------------------------------------------------------
// UI UPDATE
// -------------------------------------------------------------
function updateResults(p) {
  document.getElementById("calories").textContent = p.calories;
  document.getElementById("fat").textContent = p.fat;
  document.getElementById("carbs").textContent = p.carbs;
  document.getElementById("protein").textContent = p.protein;
}

function updateDailyStats(p) {
  dailyStats.calories += p.calories;
  dailyStats.fat += p.fat;
  dailyStats.carbs += p.carbs;
  dailyStats.protein += p.protein;

  mealHistory.push({
    time: new Date().toLocaleTimeString(),
    calories: p.calories,
    protein: p.protein,
    fat: p.fat,
    carbs: p.carbs
  });

  renderDailyProgress();
}

function renderDailyProgress() {
  const box = document.getElementById("daily-progress");

  if (!dailyPlan) {
    box.textContent = "Enter your profile to see daily stats.";
    return;
  }

  box.innerHTML = `
    <p>Calories: ${Math.round(dailyStats.calories)} / ${Math.round(dailyPlan.tdee)}</p>
    <p>Protein: ${Math.round(dailyStats.protein)} / ${Math.round(dailyPlan.protein)} g</p>
    <p>Fat: ${Math.round(dailyStats.fat)} / ${Math.round(dailyPlan.fat)} g</p>
    <p>Carbs: ${Math.round(dailyStats.carbs)} / ${Math.round(dailyPlan.carbs)} g</p>
  `;
}

// -------------------------------------------------------------
// SMART TIPS
// -------------------------------------------------------------
function updateTips() {
  const tips = document.getElementById("tips");

  if (!dailyPlan) {
    tips.textContent = "";
    return;
  }

  let arr = [];

  if (dailyStats.calories > dailyPlan.tdee)
    arr.push("⚠️ You exceeded your daily calorie limit!");

  if (dailyStats.protein < dailyPlan.protein * 0.6)
    arr.push("🍗 Add more protein sources (eggs, chicken, cottage cheese).");

  if (dailyStats.carbs < dailyPlan.carbs * 0.5)
    arr.push("🍚 Low carbs — add grains, fruits or rice.");

  if (dailyStats.fat < dailyPlan.fat * 0.5)
    arr.push("🥑 You need more healthy fats (nuts, avocado, fish).");

  tips.innerHTML = arr.length ? arr.join("<br>") : "👌 Perfect balance today!";
}

// -------------------------------------------------------------
// ANALYTICS — CHARTS
// -------------------------------------------------------------
let calorieChart = null;
let macroChart = null;

function initCharts() {
  const ctx1 = document.getElementById("calorieChart");
  const ctx2 = document.getElementById("macroChart");

  calorieChart = new Chart(ctx1, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Calories per meal",
        data: [],
        borderWidth: 2,
        borderColor: "#22c55e"
      }]
    },
    options: { responsive: true, plugins: { legend: { display: false } } }
  });

  macroChart = new Chart(ctx2, {
    type: "bar",
    data: {
      labels: ["Protein", "Fat", "Carbs"],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ["#3b82f6", "#f43f5e", "#f59e0b"]
      }]
    },
    options: { responsive: true }
  });
}

function updateCharts(p) {
  calorieChart.data.labels.push(mealHistory.length);
  calorieChart.data.datasets[0].data.push(p.calories);
  calorieChart.update();

  macroChart.data.datasets[0].data = [
    dailyStats.protein,
    dailyStats.fat,
    dailyStats.carbs
  ];
  macroChart.update();
}

// -------------------------------------------------------------
// MEAL HISTORY
// -------------------------------------------------------------
function renderHistory() {
  const list = document.getElementById("meal-history");
  list.innerHTML = mealHistory
    .map(m => `<li>${m.time}: ${m.calories} kcal (P:${m.protein}, F:${m.fat}, C:${m.carbs})</li>`)
    .join("");
}

// -------------------------------------------------------------
// EXPORT
// -------------------------------------------------------------
function exportCSV() {
  let csv = "time,calories,protein,fat,carbs\n";

  mealHistory.forEach(m =>
    csv += `${m.time},${m.calories},${m.protein},${m.fat},${m.carbs}\n`
  );

  downloadFile("nutrition_data.csv", csv);
}

function exportJSON() {
  const data = JSON.stringify(mealHistory, null, 2);
  downloadFile("nutrition_history.json", data);
}

function downloadFile(filename, content) {
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
}
