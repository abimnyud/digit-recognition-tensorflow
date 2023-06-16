/**
 * Ukuran image mnist
 */
const IMAGE_SIZE = 784;
/**
 * Menentukan banyanya jumlah kelas (zero-nine)
 */
const NUM_CLASSES = 10;
/**
 * Menentukan element pada dataset yang akan digunakan
 */
const NUM_DATASET_ELEMENTS = 65000;
/**
 * Menentukan berapa banyak elemen yang digunakan untuk training
 */
const NUM_TRAIN_ELEMENTS = 55000;
/**
 * Menghitung sisanya untuk dijadikan sebagai elemen testing
 */
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

/**
 * URL gambar dari MNIST dataset
 */
const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
/**
 * URL label dari MNIST dataset
 */
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * Class yang melakukan fetch data label dan gambar dari MNIST dataset kemudian
 * mengembalikannya dalam batch acak
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  /**
   * Memuat gambar dari MNIST dataset
   */
  async load() {
    /**
     * Membuat request ke image MNIST spritted dataset
     */
    const img = new Image();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = "";
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4
        );

        /**
         * Menentukan ukuran chunk dan juga ukuran image canvas untuk example data
         */
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize
          );
          ctx.drawImage(
            img,
            0,
            i * chunkSize,
            img.width,
            chunkSize,
            0,
            0,
            img.width,
            chunkSize
          );

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    /**
     * Membuat request ke label MNIST dataset
     */
    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] = await Promise.all([
      imgRequest,
      labelsRequest,
    ]);

    /**
     * Konversi label dataset ke Uint8Array
     */
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    /**
     * Mengacak index train/test set agar ketika melakukan select atau menarik next batch
     * akan mengembalikan data yang acak untuk training/validation.
     */
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    /**
     * Membagi gambar dan label menjadi train dan test sets.
     */
    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS
    );
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
    this.testLabels = this.datasetLabels.slice(
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
  }

  /**
   * Untuk mengambil train batch berikutnya
   */
  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  /**
   * Untuk mengambil test batch berikutnya
   */
  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  /**
   * Untuk mengambil training batch berikutnya
   */
  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image = data[0].slice(
        idx * IMAGE_SIZE,
        idx * IMAGE_SIZE + IMAGE_SIZE
      );
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      );
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }
}