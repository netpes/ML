// async function loadModel() {
//     const dataX0 = [[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.73,0.62,0.59,0.24,0.14,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.87,1.00,1.00,1.00,1.00,0.95,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.67,0.20,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.26,0.45,0.28,0.45,0.64,0.89,1.00,0.88,1.00,1.00,1.00,0.98,0.90,1.00,1.00,0.55,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.07,0.26,0.05,0.26,0.26,0.26,0.23,0.08,0.93,1.00,0.42,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.99,0.82,0.07,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.09,0.91,1.00,0.33,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.51,1.00,0.93,0.17,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.23,0.98,1.00,0.24,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.52,1.00,0.73,0.02,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.04,0.80,0.97,0.23,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.49,1.00,0.71,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.29,0.98,0.94,0.22,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.07,0.87,1.00,0.65,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.01,0.80,1.00,0.86,0.14,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,1.00,1.00,0.30,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.12,0.88,1.00,0.45,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.52,1.00,1.00,0.20,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.24,0.95,1.00,1.00,0.20,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.47,1.00,1.00,0.86,0.16,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.47,1.00,0.81,0.07,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
//     ]];
//     const myTensor = tf.tensor(dataX0);
//     const model = await tf.loadGraphModel('http://localhost:8000/TFJS/model.json')
//     const result = model.predict(myTensor)
//     result.print()
//
//     // console.log(result)
//     // console.log(model)
// }
async function predictMe(tensor){

    const model = await tf.loadGraphModel('http://localhost:8000/TFJS/model.json')
    const result = model.predict(tensor)
    result.print()
    return result
}

async function predictImage() {
    console.log('processing...')

    let image = cv.imread(canvas)

    cv.cvtColor(image, image, cv.COLOR_RGB2GRAY, 0)
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY)
    let contours = new cv.MatVector()
    let hierarchy = new cv.Mat()
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    let cnt = contours.get(0)
    let rect = cv.boundingRect(cnt)
    image = image.roi(rect)

    let height = image.rows
    let width = image.cols
    if (height > width){
        height = 20
        width = Math.round(width/(image.rows/height))
    } else {
        width=20
        height = Math.round(height/(image.cols/width))

    }
    let dsize = new cv.Size(width,height)
    cv.resize(image, image, dsize, 0,0,cv.INTER_AREA)

    const LEFT = Math.ceil(4+ (20-width)/2)
    const RIGHT = Math.floor(4+ (20-width)/2)
    const TOP = Math.ceil(4+ (20-height)/2)
    const BOTTOM = Math.floor(4+ (20-height)/2)


    let black =  new cv.Scalar(0, 0, 0, 255);
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, black);

    // center of mass
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false)
    const cx = Moments.m10 / Moments.m00
    const cy = Moments.m01 / Moments.m00
    const XSHIFT = Math.round(image.cols/2 - cx)
    const YSHIFT = Math.round(image.rows/2 - cy)
    let newSize = new cv.Size(image.cols,image.rows)
    let M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, XSHIFT, 0, 1, YSHIFT]);


// You can try more different parameters
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT,black);





    let pixelValues = image.data
    pixelValues = Float32Array.from(pixelValues)
    pixelValues = pixelValues.map((singleValue)=>{
        return singleValue /255.0
    })


    const X = tf.tensor([pixelValues])
    const model = await tf.loadGraphModel('http://localhost:8000/TFJS/model.json')
    const result = model.predict(X)
    result.print()

    // const outputCanvas = document.createElement('CANVAS')
    // cv.imshow(outputCanvas, image)
    // document.body.appendChild(outputCanvas)
    const output = result.dataSync()[0]



    //  cleanup

    image.delete();
    contours.delete();
    M.delete()
    cnt.delete();
    hierarchy.delete();
    X.dispose()
    result.dispose()
    return output
}