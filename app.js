const NUTRITION_LABELS = ["Calories", "Protein", "Fat", "Carbs"];

class NutritionAIApp {
    constructor() {
        this.dataLoader = new NutritionDataLoader();
        this.model = null;
        this.trainData = null;
        this.testData = null;
        this.isTraining = false;

        this.initializeUI();
        this.initializeBackend();
    }

    async initializeBackend() {
        try {
            await tf.setBackend("webgl");
            this.showStatus("Using WebGL backend");
        } catch {
            await tf.setBackend("cpu");
            this.showStatus("WebGL unavailable → using CPU");
        }
    }

    initializeUI() {
        document.getElementById("loadDataBtn")
            .addEventListener("click", () => this.onLoadData());

        document.getElementById("trainBtn")
            .addEventListener("click", () => this.onTrain());

        document.getElementById("evaluateBtn")
            .addEventListener("click", () => this.onEvaluate());

        document.getElementById("testFiveBtn")
            .addEventListener("click", () => this.onTestFive());

        document.getElementById("saveModelBtn")
            .addEventListener("click", () => this.onSaveDownload());

        document.getElementById("resetBtn")
            .addEventListener("click", () => this.onReset());
    }

    async onLoadData() {
        try {
            const trainFile = document.getElementById("trainFile").files[0];
            const testFile = document.getElementById("testFile").files[0];

            if (!trainFile || !testFile) {
                this.showError("Select both train and test CSVs!");
                return;
            }

            this.showStatus("Loading train data...");
            this.trainData = await this.dataLoader.loadTrainFromFiles(trainFile);

            this.showStatus("Loading test data...");
            this.testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.updateDataStatus(
                this.trainData.count,
                this.testData.count
            );

            this.showStatus("Rendering EDA...");
            await this.renderEDA();

            this.showStatus("Data loaded!");
        } catch (err) {
            this.showError(err.message);
        }
    }

    createModel() {
        const model = tf.sequential();

        model.add(tf.layers.conv2d({
            inputShape: [224, 224, 3],
            filters: 32,
            kernelSize: 5,
            activation: "relu"
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: "relu"
        }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 256, activation: "relu" }));
        model.add(tf.layers.dropout({ rate: 0.3 }));
        model.add(tf.layers.dense({ units: 128, activation: "relu" }));

        // 4 выхода: калории, белки, жиры, углеводы
        model.add(tf.layers.dense({ units: 4, activation: "linear" }));

        model.compile({
            optimizer: tf.train.adam(0.0001),
            loss: "meanSquaredError",
            metrics: ["mae"]
        });

        this.model = model;
        this.updateModelInfo();
        return model;
    }

    async onTrain() {
        if (!this.trainData) return this.showError("Load data first!");

        if (this.model) this.model.dispose();

        this.model = this.createModel();
        this.showStatus("Training...");

        this.isTraining = true;

        const history = await this.model.fit(
            this.trainData.xs,
            this.trainData.ys,
            {
                epochs: 10,
                batchSize: 16,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(
                            `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(3)}, mae=${logs.mae.toFixed(3)}`
                        );
                    }
                }
            }
        );

        this.isTraining = false;
        this.showStatus("Training completed.");
    }

    async onEvaluate() {
        if (!this.model || !this.testData) return;

        this.showStatus("Evaluating...");
        const result = await this.model.evaluate(
            this.testData.xs,
            this.testData.ys
        );

        this.showStatus(
            `Test → Loss=${result[0].dataSync()[0].toFixed(3)}, MAE=${result[1].dataSync()[0].toFixed(3)}`
        );
    }

    async onTestFive() {
        if (!this.model || !this.testData) return;

        const indices = tf.util.createShuffledIndices(this.testData.count).slice(0, 5);

        const batchXs = tf.gather(this.testData.xs, indices);
        const batchYs = tf.gather(this.testData.ys, indices);

        const preds = this.model.predict(batchXs);

        const predArr = await preds.array();
        const trueArr = await batchYs.array();

        this.showStatus("Showing 5 predictions...");

        const container = document.getElementById("previewContainer");
        container.innerHTML = "";

        for (let i = 0; i < 5; i++) {
            const div = document.createElement("div");
            div.innerHTML = `
                <b>True:</b> ${trueArr[i].map(v => v.toFixed(1)).join(", ")}<br>
                <b>Predicted:</b> ${predArr[i].map(v => v.toFixed(1)).join(", ")}
            `;
            container.appendChild(div);
        }

        batchXs.dispose();
        batchYs.dispose();
        preds.dispose();
    }

    async renderEDA() {
        const container = document.getElementById("edaContainer");
        container.innerHTML = "<h3>EDA</h3>";

        const labels = this.trainData.labels;
        const stats = [0, 1, 2, 3].map(i => labels.map(r => r[i]));

        for (let i = 0; i < 4; i++) {
            const block = document.createElement("div");
            block.innerHTML = `
                <h4>${NUTRITION_LABELS[i]}</h4>
                min=${Math.min(...stats[i]).toFixed(1)}  
                max=${Math.max(...stats[i]).toFixed(1)}  
                mean=${(stats[i].reduce((a,b)=>a+b)/stats[i].length).toFixed(1)}
            `;
            container.appendChild(block);
        }
    }

    async onSaveDownload() {
        if (!this.model) return;

        await this.model.save("downloads://nutrition-model");
        this.showStatus("Model saved.");
    }

    onReset() {
        if (this.model) this.model.dispose();
        this.dataLoader.dispose();

        this.showStatus("Reset done.");
    }

    updateModelInfo() {
        const el = document.getElementById("modelInfo");
        if (!el) return;

        if (!this.model) {
            el.innerHTML = "<p>No model loaded</p>";
            return;
        }

        el.innerHTML = `
            <p>Layers: ${this.model.layers.length}</p>
            <p>Params: ${this.model.countParams()}</p>
            <p>Outputs: Regression (4)</p>
        `;
    }

    updateDataStatus(train, test) {
        const el = document.getElementById("dataStatus");
        el.innerHTML = `
            <p>Train samples: ${train}</p>
            <p>Test samples: ${test}</p>
        `;
    }

    showStatus(msg) {
        const logs = document.getElementById("trainingLogs");
        logs.textContent += `[info] ${msg}\n`;
        logs.scrollTop = logs.scrollHeight;
    }

    showError(msg) {
        const logs = document.getElementById("trainingLogs");
        logs.textContent += `[error] ${msg}\n`;
        logs.scrollTop = logs.scrollHeight;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    new NutritionAIApp();
});
