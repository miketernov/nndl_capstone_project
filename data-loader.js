class NutritionDataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (event) => {
                try {
                    const lines = event.target.result
                        .split("\n")
                        .filter(l => l.trim() !== "");

                    const imagePaths = [];
                    const labels = [];

                    for (const line of lines) {
                        const row = line.split(",");

                        if (row.length !== 5) continue;

                        const [imgPath, cal, pr, fat, carb] = row;

                        imagePaths.push(imgPath.trim());
                        labels.push([
                            parseFloat(cal),
                            parseFloat(pr),
                            parseFloat(fat),
                            parseFloat(carb)
                        ]);
                    }

                    if (imagePaths.length === 0) {
                        reject(new Error("No valid nutrition entries found"));
                        return;
                    }

                    // Загружаем изображения → тензоры
                    const images = [];
                    for (let p of imagePaths) {
                        const tensor = await this.loadImageTensor("images/" + p);
                        images.push(tensor);
                    }

                    const xs = tf.stack(images).div(255);
                    const ys = tf.tensor2d(labels);

                    resolve({
                        xs,
                        ys,
                        count: xs.shape[0],
                        labels,
                        imagePaths
                    });
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () =>
                reject(new Error("Failed to read CSV file"));
            reader.readAsText(file);
        });
    }

    async loadImageTensor(url) {
        const img = new Image();
        img.src = url;

        await new Promise(res => {
            img.onload = res;
            img.onerror = () => res();
        });

        return tf.tidy(() => tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]));
    }

    async loadTrainFromFiles(file) {
        this.trainData = await this.loadCSV(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        this.testData = await this.loadCSV(file);
        return this.testData;
    }

    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
