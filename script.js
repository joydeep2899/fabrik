

const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');
var model=undefined;
const relu='relu';

 model=tf.sequential();
model.add(tf.layers.conv2d({filters:64,kernelSize:3,inputShape:[300,300,3],activation:'relu'}  ))

model.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:relu }   ))
model.add(tf.layers.maxPooling2d({poolSize:2}))
model.add(tf.layers.conv2d({filters:128,kernelSize:3,activation:relu }   ))
model.add(tf.layers.maxPooling2d({poolSize:2}))
model.add(tf.layers.conv2d({filters:256,kernelSize:3,activation:relu }   ))
model.add(tf.layers.maxPooling2d({poolSize:35}))
model.add(tf.layers.maxPooling2d({poolSize:2}))

model.add(tf.layers.dense({units:5,activation:'relu'}  ))


const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia);
  }
  

  if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
  
  
function enableCam(event) {
    
    if (!model) {
      return;
    }
    
    event.target.classList.add('removed');  
    
  
    const constraints = {
      video: true
    };
  
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      video.srcObject = stream;
      video.addEventListener('loadeddata', predictWebcam);
     
    });
  }
  

  /*
 
  cocoSsd.load().then(function (loadedModel) {
    //model = loadedModel;
   
    demosSection.classList.remove('invisible');
  });
*/


var predictions;
  
  var children = [];

async function predictWebcam() {
  var cam = await tf.data.webcam(video);
  var  img = await cam.capture();
 // img=img.expandDims(0)

   img=tf.image.resizeNearestNeighbor(img,[300,300])
   img=img.expandDims(0)
  
 predictions= model.predict(img)


    console.log(predictions[0])
    
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    children.splice(0);

    for (let n = 0; n < predictions.length; n++) {
      
      if (predictions[n].score > 0.66) {




       
const x = predictions[n].bbox[0];
const y = predictions[n].bbox[1];
const width = predictions[n].bbox[2];
const height = predictions[n].bbox[3];

ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

const font = "13px sans-serif";
ctx.font = font;
ctx.textBaseline = "top";
predictions.forEach(prediction => {
  const x = prediction.bbox[0];
  const y = prediction.bbox[1];
  const width = prediction.bbox[2];
  const height = prediction.bbox[3];
 
  ctx.strokeStyle = "#00FFFF";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, (width-x)/4, (height-y)/4);
 
  ctx.fillStyle = "#00FFFF";
  const textWidth = ctx.measureText(prediction.class).width;
  const textHeight = parseInt(font, 10); // base 10
  ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
});

predictions.forEach(prediction => {
  const x = prediction.bbox[0];
  const y = prediction.bbox[1];
  
  ctx.fillStyle = "#000000";
  ctx.fillText(prediction.class, x, y);
});







      }
    }
    
  
    window.requestAnimationFrame(predictWebcam);
  }
