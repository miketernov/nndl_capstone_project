// -------------------------------------------------------------
// GLOBALS & CONSTANTS
// -------------------------------------------------------------
let model = null;

// Nutrition normalization params
const NUTRITION_MEAN = [257.7, 214.42, 12.97, 19.28, 18.26];
const NUTRITION_STD  = [211.42, 153.17, 13.72, 16.17, 20.14];

// State
let dailyStats = { calories: 0, protein: 0, fat: 0, carbs: 0 };
let dailyPlan  = null;
let mealHistory = [];

let mealTypeStats = {
  breakfast: {cal:0,p:0,f:0,c:0},
  lunch:     {cal:0,p:0,f:0,c:0},
  dinner:    {cal:0,p:0,f:0,c:0},
  snack:     {cal:0,p:0,f:0,c:0},
};

// LocalStorage keys
const PROFILE_KEY   = "nutritionAI_profile";
const STATE_KEY     = "nutritionAI_state";
const LAST_MEAL_KEY = "nutritionAI_lastMealTs";

// Notification flags
let notifCaloriesExceededSent = false;

// Charts
let calorieChart = null;
let macroChart   = null;
let mealTypeChart = null;

// -------------------------------------------------------------
// THEME
// -------------------------------------------------------------
function toggleTheme() {
  document.documentElement.classList.toggle("light");
}

// -------------------------------------------------------------
// HELP FIX NOTIFICATIONS
// -------------------------------------------------------------
function helpFixNotifications() {
  alert(
    "To enable notifications:\n" +
    "1. Click the 🔒 icon next to the address bar.\n" +
    "2. Select 'Site settings'.\n" +
    "3. Find 'Notifications' and set it to 'Allow'."
  );
}

// -------------------------------------------------------------
// MODEL LOADING
// -------------------------------------------------------------
function getModelUrl() {
  const base = window.location.href.replace(/index\.html?$/, "");
  return new URL("model.json", base).toString();
}

window.addEventListener("load", async () => {
  const status = document.getElementById("status");

  try {
    status.textContent = "Loading model...";
    model = await tf.loadGraphModel(getModelUrl());
    status.textContent = "Model loaded.";
  } catch (e) {
    console.error("Model load error:", e);
    status.textContent = "Failed to load model.";
  }

  const fileInput = document.getElementById("file-input");
  fileInput.addEventListener("change", handleFileChange);

  initCharts();
  initMealTypeChart();

  loadProfileFromStorage();
  loadStateFromStorage();
  rebuildUIFromState();
  checkLongNoMealNotification();

  notifCaloriesExceededSent = false;
});

// -------------------------------------------------------------
// PROFILE PLAN
// -------------------------------------------------------------
function calculatePlan() {
  const username  = document.getElementById("username").value.trim();
  const age       = +document.getElementById("age").value;
  const weight    = +document.getElementById("weight").value;
  const height    = +document.getElementById("height").value;
  const activity  = +document.getElementById("activity").value;
  const goal      = document.getElementById("goal").value;

  if (!age || !weight || !height) {
    alert("Please fill out age, weight and height.");
    return;
  }

  // Mifflin-St Jeor (male baseline)
  let bmr  = 10 * weight + 6.25 * height - 5 * age + 5;
  let tdee = bmr * activity;

  if (goal === "loss")  tdee *= 0.82;
  if (goal === "gain")  tdee *= 1.15;

  const protein = weight * 1.8;
  const fat     = weight * 0.9;
  const carbs   = (tdee - protein * 4 - fat * 9) / 4;

  dailyPlan = { tdee, protein, fat, carbs };

  const rec = document.getElementById("recommendations");

  rec.innerHTML = `
    <p>👋 Welcome${username ? ", <b>" + username + "</b>" : ""}!</p>
    <p><b>Daily calories:</b> ${Math.round(tdee)} kcal</p>
    <p><b>Protein:</b> ${Math.round(protein)} g</p>
    <p><b>Fat:</b> ${Math.round(fat)} g</p>
    <p><b>Carbs:</b> ${Math.round(carbs)} g</p>
  `;

  saveProfileToStorage();
  ensureNotificationPermission();

  notifCaloriesExceededSent = false;

  renderDailyProgress();
  updateTips();
  renderSummary();
}

// -------------------------------------------------------------
// MEAL TYPE
// -------------------------------------------------------------
function getSelectedMealType() {
  const radios = document.querySelectorAll('input[name="mealType"]');
  for (let r of radios) if (r.checked) return r.value;
  return "unknown";
}

function mealTypeToLabel(type) {
  return {
    breakfast: "Breakfast",
    lunch: "Lunch",
    dinner: "Dinner",
    snack: "Snack"
  }[type] || "Meal";
}

// -------------------------------------------------------------
// IMAGE HANDLING
// -------------------------------------------------------------
function handleFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (!model) {
    alert("Model not loaded yet.");
    return;
  }

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
      renderSummary();

      saveStateToStorage();
      saveLastMealTimestamp();

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
// UI UPDATE
// -------------------------------------------------------------
function updateResults(p) {
  document.getElementById("calories").textContent = p.calories;
  document.getElementById("fat").textContent      = p.fat;
  document.getElementById("carbs").textContent    = p.carbs;
  document.getElementById("protein").textContent  = p.protein;
}

function updateDailyStats(p) {
  dailyStats.calories += p.calories;
  dailyStats.fat      += p.fat;
  dailyStats.carbs    += p.carbs;
  dailyStats.protein  += p.protein;

  const type = getSelectedMealType();

  mealHistory.push({
    time: new Date().toLocaleTimeString(),
    ts:   Date.now(),
    mealType: type,
    calories: p.calories,
    protein:  p.protein,
    fat:      p.fat,
    carbs:    p.carbs
  });

  renderDailyProgress();

  if (dailyPlan && dailyStats.calories > dailyPlan.tdee && !notifCaloriesExceededSent) {
    sendNotification("you exceeded your daily calorie target.");
    notifCaloriesExceededSent = true;
  }
}

function updateMealTypeStats(p) {
  const type = getSelectedMealType();
  if (!mealTypeStats[type]) {
    mealTypeStats[type] = {cal:0,p:0,f:0,c:0};
  }
  mealTypeStats[type].cal += p.calories;
  mealTypeStats[type].p   += p.protein;
  mealTypeStats[type].f   += p.fat;
  mealTypeStats[type].c   += p.carbs;
}

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
// TIPS & NOTIFICATIONS
// -------------------------------------------------------------
function updateTips() {
  const tipsEl = document.getElementById("tips");
  if (!dailyPlan) {
    tipsEl.textContent = "";
    return;
  }

  const msgs = [];

  if (dailyStats.calories > dailyPlan.tdee)
    msgs.push("⚠️ You exceeded your daily calorie limit.");

  if (dailyStats.protein < dailyPlan.protein * 0.4)
    msgs.push("🍗 Too little protein — add chicken or eggs.");

  if (dailyStats.carbs < dailyPlan.carbs * 0.4)
    msgs.push("🍚 Low carbs — add rice, bread or fruit.");

  if (dailyStats.fat < dailyPlan.fat * 0.4)
    msgs.push("🥑 Low healthy fats — add nuts or avocado.");

  tipsEl.innerHTML = msgs.length ? msgs.join("<br>") : "👌 Balanced day!";
}

// -------------------------------------------------------------
// SUMMARY BLOCK
// -------------------------------------------------------------
function renderSummary() {
  const el = document.getElementById("summary");
  if (!el) return;

  if (mealHistory.length === 0) {
    el.textContent = "No meals logged yet.";
    return;
  }

  const totalMeals = mealHistory.length;
  const totalCalories = mealHistory.reduce((s,m) => s + m.calories, 0);

  let typeTotals = { breakfast:0, lunch:0, dinner:0, snack:0 };
  mealHistory.forEach(m => {
    if (typeTotals[m.mealType] !== undefined) {
      typeTotals[m.mealType] += m.calories;
    }
  });

  let favType = "—";
  let maxCal = 0;
  for (const [k,v] of Object.entries(typeTotals)) {
    if (v > maxCal) {
      maxCal = v;
      favType = mealTypeToLabel(k);
    }
  }

  const avgCalories = Math.round(totalCalories / totalMeals);

  el.innerHTML = `
    <p>Total meals logged: <b>${totalMeals}</b></p>
    <p>Total calories (all time): <b>${totalCalories}</b> kcal</p>
    <p>Average calories per meal: <b>${avgCalories}</b> kcal</p>
    <p>Most caloric meal type: <b>${favType}</b></p>
  `;
}

// -------------------------------------------------------------
// CHARTS (with datalabels)
// -------------------------------------------------------------
function initCharts() {
  const ctx1 = document.getElementById("calorieChart");
  const ctx2 = document.getElementById("macroChart");

  calorieChart = new Chart(ctx1, {
    type: "line",
    plugins: [ChartDataLabels],
    data: {
      labels: [],
      datasets: [{
        label: "Calories per meal",
        data: [],
        borderWidth: 2,
        borderColor: "#22c55e",
        pointRadius: 4,
        tension: 0.2
      }]
    },
    options: {
      plugins: {
        datalabels: {
          align: "top",
          anchor: "center",
          color: "#fff",
          formatter: (_v, ctx) => ctx.chart.data.labels[ctx.dataIndex]
        }
      }
    }
  });

  macroChart = new Chart(ctx2, {
    type: "bar",
    plugins: [ChartDataLabels],
    data: {
      labels: ["Protein", "Fat", "Carbs"],
      datasets: [{
        label: "Macros (g)",
        data: [0,0,0],
        backgroundColor: ["#3b82f6", "#ef4444", "#f59e0b"]
      }]
    },
    options: {
      plugins: {
        datalabels: {
          anchor: "end",
          align: "top",
          color: "#fff",
          formatter: v => Math.round(v)
        }
      }
    }
  });
}

function updateCharts(p) {
  const lastMeal = mealHistory[mealHistory.length - 1];
  calorieChart.data.labels.push(mealTypeToLabel(lastMeal.mealType));
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
    plugins: [ChartDataLabels],
    data: {
      labels: ["Breakfast", "Lunch", "Dinner", "Snack"],
      datasets: [{
        label: "Calories",
        data: [0,0,0,0],
        backgroundColor: ["#3b82f6","#10b981","#f59e0b","#ef4444"]
      }]
    },
    options: {
      plugins: {
        datalabels: {
          anchor: "end",
          align: "top",
          color: "#fff",
          formatter: v => Math.round(v)
        }
      }
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
      <li><b>${mealTypeToLabel(m.mealType)}</b> — ${m.time}: 
      ${m.calories} kcal (P:${m.protein}, F:${m.fat}, C:${m.carbs})</li>
    `)
    .join("");
}

// -------------------------------------------------------------
// EXPORT
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
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// -------------------------------------------------------------
// LOCALSTORAGE: PROFILE & STATE
// -------------------------------------------------------------
function saveProfileToStorage() {
  if (typeof localStorage === "undefined") return;

  const username = document.getElementById("username").value.trim();

  const profile = {
    username: username || null,
    age: document.getElementById("age").value,
    weight: document.getElementById("weight").value,
    height: document.getElementById("height").value,
    activity: document.getElementById("activity").value,
    goal: document.getElementById("goal").value
  };

  try {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  } catch (_) {}
}

function loadProfileFromStorage() {
  if (typeof localStorage === "undefined") return;

  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (!raw) return;

    const p = JSON.parse(raw);

    if (p.username !== null && p.username !== "")
      document.getElementById("username").value = p.username;

    if (p.age)      document.getElementById("age").value = p.age;
    if (p.weight)   document.getElementById("weight").value = p.weight;
    if (p.height)   document.getElementById("height").value = p.height;
    if (p.activity) document.getElementById("activity").value = p.activity;
    if (p.goal)     document.getElementById("goal").value = p.goal;

    calculatePlan();
  } catch (_) {}
}

function saveStateToStorage() {
  if (typeof localStorage === "undefined") return;
  const state = {
    dailyStats,
    mealHistory,
    mealTypeStats
  };
  try {
    localStorage.setItem(STATE_KEY, JSON.stringify(state));
  } catch (_) {}
}

function loadStateFromStorage() {
  if (typeof localStorage === "undefined") return;

  try {
    const raw = localStorage.getItem(STATE_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);

    if (s.dailyStats)    dailyStats    = s.dailyStats;
    if (s.mealHistory)   mealHistory   = s.mealHistory;
    if (s.mealTypeStats) mealTypeStats = s.mealTypeStats;
  } catch (_) {}
}

function rebuildUIFromState() {
  renderDailyProgress();
  renderHistory();
  rebuildChartsFromState();
  renderSummary();
  updateTips();
}

function rebuildChartsFromState() {
  if (!calorieChart || !macroChart || !mealTypeChart) return;

  // calorieChart
  calorieChart.data.labels = [];
  calorieChart.data.datasets[0].data = [];
  mealHistory.forEach(m => {
    calorieChart.data.labels.push(mealTypeToLabel(m.mealType));
    calorieChart.data.datasets[0].data.push(m.calories);
  });
  calorieChart.update();

  // macroChart
  macroChart.data.datasets[0].data = [
    dailyStats.protein,
    dailyStats.fat,
    dailyStats.carbs
  ];
  macroChart.update();

  // mealTypeChart
  mealTypeChart.data.datasets[0].data = [
    mealTypeStats.breakfast.cal,
    mealTypeStats.lunch.cal,
    mealTypeStats.dinner.cal,
    mealTypeStats.snack.cal
  ];
  mealTypeChart.update();
}

function saveLastMealTimestamp() {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(LAST_MEAL_KEY, String(Date.now()));
  } catch (_) {}
}

// -------------------------------------------------------------
// NOTIFICATIONS
// -------------------------------------------------------------
function ensureNotificationPermission() {
  if (typeof Notification === "undefined") return;

  if (Notification.permission === "default") {
    Notification.requestPermission();
  }
}

function sendNotification(message) {
  if (typeof Notification === "undefined") return;
  if (Notification.permission !== "granted") return;

  const username = document.getElementById("username").value.trim() || "User";

  try {
    new Notification(`${username}, ${message}`);
  } catch (_) {}
}

function checkLongNoMealNotification() {
  if (typeof localStorage === "undefined") return;
  if (typeof Notification === "undefined") return;

  try {
    const raw = localStorage.getItem(LAST_MEAL_KEY);
    if (!raw) return;

    const last = Number(raw);
    if (!last) return;

    const now = Date.now();
    const diffHours = (now - last) / (1000 * 60 * 60);

    if (diffHours > 4 && Notification.permission === "granted") {
      sendNotification("you haven't logged a meal in over 4 hours.");
    }
  } catch (_) {}
}
