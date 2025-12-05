// -------------------------------------------------------------
// GLOBAL VARIABLES
// -------------------------------------------------------------
let model = null;

const NUTRITION_MEAN = [257.7, 214.42, 12.97, 19.28, 18.26];
const NUTRITION_STD = [211.42, 153.17, 13.72, 16.17, 20.14];

let dailyStats = { calories: 0, protein: 0, fat: 0, carbs: 0 };
let dailyPlan = null;
let mealHistory = [];

let mealTypeStats = {
  breakfast: {cal:0,p:0,f:0,c:0},
  lunch:     {cal:0,p:0,f:0,c:0},
  dinner:    {cal:0,p:0,f:0,c:0},
  snack:     {cal:0,p:0,f:0,c:0},
};

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
    s.textContent = "Model loaded.";
  } catch (e) {
    console.error("Model error:", e);
    s.textContent = "Failed to load model.";
  }

  document.getElementById("file-input").addEventListener("change", handleFileChange);

  initCharts();
  initMealTypeChart();
});

// -------------------------------------------------------------
// PROFILE PLAN
// -------------------------------------------------------------
function calculatePlan() {
  const age = +document.getElementById("age").value;
  const weight = +document.getElementById("weight").value;
  const height = +document.getElementById("height").value;
  const activity = +document.getElementById("activity").value;
  const goal = document.getElementById("goal").value;

  if (!age || !weight || !height) {
    alert("Please fill out all fields.");
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
// MEAL TYPE GETTER
// -------------------------------------------------------------
function getSelectedMealType() {
  const radios = document.querySelectorAll('input[name="mealType"]');
  for (let r of radios) if (r.checked) return r.value;
  return "unknown";
}

// -------------------------------------------------------------
// IMAGE HANDLING
// -------------------------------------------------------------
function handleFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    const img = document.getElementById("preview");

    img.onload = async () => {
      document.getElementById("status").textContent = "Predicting...";

      const pred = await runInferenceOnImage(img);

      updateResults(pred);
      updateDailyStats(pred);
      updateCharts(pred);
      updateMealTypeStats(pred);
      updateMealTypeChart();
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

    const d = vals.map((v, i) => v * NUTRITION_STD[i] + NUTRITION_MEAN[i]);

    return {
      calories: Math.round(d[0]),
      mass: d[1],
      fat: Math.round(d[2]),
      carbs: Math.round(d[3]),
      protein: Math.round(d[4])
    };
  });
}

// -------------------------------------------------------------
// UPDATE UI
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

  const type = getSelectedMealType();

  mealHistory.push({
    time: new Date().toLocaleTimeString(),
    mealType: type,
    calories: p.calories,
    protein: p.protein,
    fat: p.fat,
    carbs: p.carbs
  });

  renderDailyProgress();
}

// -------------------------------------------------------------
// MEAL TYPE STATS
// -------------------------------------------------------------
function updateMealTypeStats(p) {
  const type = getSelectedMealType();
  mealTypeStats[type].cal += p.calories;
  mealTypeStats[type].p += p.protein;
  mealTypeStats[type].f += p.fat;
  mealTypeStats[type].c += p.carbs;
}

// -------------------------------------------------------------
// PROGRESS
// -------------------------------------------------------------
function renderDailyProgress() {
  const box = document.getElementById("daily-progress");

  if (!dailyPlan) {
    box.textContent = "Set your profile to see progress.";
    return;
  }

  box.innerHTML = `
    <p>Calories: ${dailyStats.calories} / ${Math.round(dailyPlan.tdee)}</p>
    <p>Protein: ${dailyStats.protein} / ${Math.round(dailyPlan.protein)} g</p>
    <p>Fat: ${dailyStats.fat} / ${Math.round(dailyPlan.fat)} g</p>
    <p>Carbs: ${dailyStats.carbs} / ${Math.round(dailyPlan.carbs)} g</p>
  `;
}

// -------------------------------------------------------------
// TIPS / RECOMMENDATIONS
// -------------------------------------------------------------
function updateTips() {
  const tips = document.getElementById("tips");
  if (!dailyPlan) {
    tips.textContent = "";
    return;
  }

  let msg = [];

  if (dailyStats.calories > dailyPlan.tdee)
    msg.push("⚠️ You exceeded your daily calorie limit!");

  if (dailyStats.protein < dailyPlan.protein * 0.4)
    msg.push("🍗 Too little protein — add eggs, chicken or cottage cheese.");

  if (dailyStats.carbs < dailyPlan.carbs * 0.4)
    msg.push("🍚 Low carbs — add rice, pasta or fruit.");

  if (dailyStats.fat < dailyPlan.fat * 0.4)
    msg.push("🥑 Increase healthy fats — nuts, avocado, olive oil.");

  tips.innerHTML = msg.length ? msg.join("<br>") : "👌 Balanced day so far.";
}

// -------------------------------------------------------------
// CHARTS
// -------------------------------------------------------------
let calorieChart = null;
let macroChart = null;
let mealTypeChart = null;

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
    }
  });

  macroChart = new Chart(ctx2, {
    type: "bar",
    data: {
      labels: ["Protein", "Fat", "Carbs"],
      datasets: [{
        data: [0,0,0],
        backgroundColor: ["#3b82f6", "#ef4444", "#f59e0b"]
      }]
    }
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
// MEAL TYPE CHART
// -------------------------------------------------------------
function initMealTypeChart() {
  const ctx = document.getElementById("mealTypeChart");

  mealTypeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Breakfast", "Lunch", "Dinner", "Snack"],
      datasets: [{
        label: "Calories",
        data: [0,0,0,0],
        backgroundColor: ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
      }]
    }
  });
}

function updateMealTypeChart() {
  mealTypeChart.data.datasets[0].data = [
    mealTypeStats.breakfast.cal,
    mealTypeStats.lunch.cal,
    mealTypeStats.dinner.cal,
    mealTypeStats.snack.cal
  ];
  mealTypeChart.update();
}

// -------------------------------------------------------------
// HISTORY
// -------------------------------------------------------------
function renderHistory() {
  const list = document.getElementById("meal-history");

  list.innerHTML = mealHistory
    .map(m => `
      <li><b>${m.mealType.toUpperCase()}</b> — ${m.time}: 
      ${m.calories} kcal (P:${m.protein}, F:${m.fat}, C:${m.carbs})</li>
    `)
    .join("");
}

// -------------------------------------------------------------
// EXPORT FUNCTIONS
// -------------------------------------------------------------
function exportCSV() {
  let csv = "time,mealType,calories,protein,fat,carbs\n";

  mealHistory.forEach(m =>
    csv += `${m.time},${m.mealType},${m.calories},${m.protein},${m.fat},${m.carbs}\n`
  );

  downloadFile("nutrition_data.csv", csv);
}

function exportJSON() {
  downloadFile("nutrition_history.json", JSON.stringify(mealHistory, null, 2));
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
