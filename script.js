import { MnistData } from "./data.js";

/**
 * Untuk menampilkan contoh input gambar dari MNIST dataset
 */
async function showExamples(data) {
  /**
   * Membuat container baru di tf visor
   */
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  /**
   * Mengambil 20 contoh input gambar dari test batch
   */
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  /**
   * Membuat elemn canvas untuk me-render gambarnya di halaman website
   */
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      /**
       * Mengubah ukuran gambar menjadi 28 x 28 pixel
       */
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

/**
 * Fungsi utama untuk menjalankan semua proses training dan validation
 */
async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  const model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture", tab: "Model" }, model);

  await train(model, data);

  await showAccuracy(model, data);
  await showConfusion(model, data);
}

/**
 * Untuk menampilkan tensorflow visor ke halaman website
 */
document.addEventListener("DOMContentLoaded", run);

/**
 * Fungsi untuk melakukan training model
 */
async function train(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "Model Training",
    tab: "Model",
    styles: { height: "1000px" },
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  /**
   * Menentukan ukuran data dalam setiap batch
   */
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  /**
   * Training model
   */
  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

/**
 * Fungsi untuk mengambil model
 */
function getModel() {
  const model = tf.sequential();

  /**
   * Menentukan ukuran gambar dan channel
   */
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  /**
   * Pada layer pertama dari convolutional neural network,
   * harus menentukan input shape nya. Kemudian menentukan
   * beberapa parameter untuk operasi konvolusi pada layer ini
   */
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );

  /**
   * Lapisan MaxPooling bertindak sebagai semacam downsampling
   * menggunakan nilai maksimal di suatu wilayah.
   */
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  /**
   * Mengulangi kedua proses di atas (conv2d + maxPooling)
   * dengan tambahan lebih banyak filter dalam operasi konvolusi
   */
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  /**
   * Layer keempat ini meratakan output dari filter 2D menjadi vektor 1D
   * untuk mempersiapkannya sebagai input ke dalam lapisan terakhir.
   */
  model.add(tf.layers.flatten());

  /**
   * Layer terakhir ini adalah dense layer yang memiliki 10 unit keluaran,
   * satu untuk setiap kelas keluaran (yaitu 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
   */
  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  /**
   * Memilih optimizer, loss function dan accuracy metric,
   * kemudian di-comple dan kembalikan modelnya
   */
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

/**
 * Menentukan nama kelas
 */
const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

/**
 * Fungsi untuk melakukan prediksi
 */
function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

/**
 * Fungsi untuk menampilkan akurasi
 */
async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: "Accuracy", tab: "Evaluation" };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

/**
 * Fungsi untuk menampilkan tabel confusion matrix
 */
async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: "Confusion Matrix", tab: "Evaluation" };
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames,
  });

  labels.dispose();
}
