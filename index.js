//const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const modelname= "ssd connector"

const img_shape=[300,300,4];
var target =[],input=[]
const fs = require('fs');
//var w=tf.variable(5)
//var y=tf.variable(5)

const relu='relu';

function load_image_batch(path,dataset,start,end){
var name='';
    for(i=start;i<end;i++){
 

        if(i+1==28)
        continue;
    
        name=path+(i+1)
    
        if(i<47){
        
            name=name+'.jpg';
            buff=fs.readFileSync(name);
        dataset.push(tf.node.decodeJpeg(buff));
        console.log(i+1+'xxx');
        }else{
        
            name=name+'.png'
            buff=fs.readFileSync(name);
        dataset.push(tf.node.decodePng(buff));
        }
        
        
       
     
    
       
    
    }

    return dataset
    
}

function load_dataset(path){

const no_of_images=52;
var i=0,j=0;
var filename=[],name='';
var dataset=[];var buff;

dataset=load_image_batch(path,dataset,0,10);

dataset=load_image_batch(path,dataset,10,20);

dataset=load_image_batch(path,dataset,20,30);

//dataset=load_image_batch(path,dataset,0,10);






return dataset;



}







function conv_decode(conv_output_tensor){
// decode conv output
var conv_output=null;
const box={
    confidence:0,
    coordinates:[,,,]
}

var box1=box,box2=box,box3=box,box4=box;


conv_output= conv_output_tensor.arraySync();
  
conv_output=conv_output[0][0][0]



    box1.confidence=conv_output.slice(0,1);
    box1.coordinates=conv_output.slice(1,5);
   /* 
    box2.confidence=conv_output.slice(5,6);
    box2.coordinates=conv_output.slice(6,10);
    
    
    box3.confidence=conv_output.slice(10,11);
    box3.coordinates=conv_output.slice(11,15);
    
    
    box4.confidence=conv_output.slice(15,16);
    box4.coordinates=conv_output.slice(16,20);
    
    */




return [box1];














    
}


function loss(yTrue,yPred){

    /*
console.log("enter loss function ")


y_true=y_true.arraySync();
y_pred=y_pred.arraySync();

y_true=y_true[0][0][0];
y_pred=y_pred[0][0][0];

console.log("Y_true"+y_true);
console.log("y_pred"+y_pred);
*/
//var decode=input[i]
//var locloss=null;
//var box1=decode[0],box2=decode[1],box3=decode[2],box4=decode[3];

/*
var N=4
var i=0;
var box1_confidence,box1_x1_diff,box1_y1_diff,box1_x2_diff,box1_y2_diff;
box1_confidence=y_true[0]-y_pred[0];
box1_x1_diff=y_true[1]-y_pred[1];
box1_y1_diff=y_true[2]-y_pred[2];
box1_x2_diff=y_true[3]-y_pred[3];
box1_y2_diff=y_true[4]-y_pred[4];







var t=target[i]
box1_confidence=1-box1.confidence[0]
box1_x1_diff=target[i].x1-box1.coordinates[0]
box1_y1_diff=target[i].y1-box1.coordinates[1]
box1_x2_diff=target[i].x2-box1.coordinates[2]
box1_y2_diff=target[i].y2-box1.coordinates[3]

box2_confidence=1-box2.confidence
box2_x1_diff=target[i].x1-box2.coordinates[0]
box2_y1_diff=target[i].y1-box2.coordinates[1]
box2_x2_diff=target[i].x2-box2.coordinates[2]
box2_y2_diff=target[i].y2-box2.coordinates[3]

box3_confidence=1-box3.confidence
box3_x1_diff=target[i].x1-box3.coordinates[0]
box3_y1_diff=target[i].y1-box3.coordinates[1]
box3_x2_diff=target[i].x2-box3.coordinates[2]
box3_y2_diff=target[i].y2-box3.coordinates[3]


box4_confidence=1-box4.confidence
box4_x1_diff=target[i].x1-box4.coordinates[0]
box4_y1_diff=target[i].y1-box4.coordinates[1]
box4_x2_diff=target[i].x2-box4.coordinates[2]
box4_y2_diff=target[i].y2-box4.coordinates[3]
*/
//var alpha=0.1




//total_loss=total_loss+box1_confidence+alpha*(box1_x1_diff+box1_x2_diff+box1_y1_diff+box1_y2_diff)
//total_loss=(y_true[0]-y_pred[0])+0.1*((y_true[1]-y_pred[1])+(y_true[2]-y_pred[2])+(y_true[3]-y_pred[3])+(y_true[4]-y_pred[4]))


//console.log("total _ loss:"+total_loss)


//return tf.tensor(total_loss);
return tf.tidy(() => {
    // Scale the the first column (0-1 shape indicator) of `yTrue` in order
    // to ensure balanced contributions to the final loss value
    // from shape and bounding-box predictions.
    const LABEL_MULTIPLIER = [300, 1, 1, 1, 1];
  //  var x=tf.metrics.meanAbsoluteError(yTrue,yPred);
 







    var x=yTrue.sub(yPred);
    yTrue=yTrue.mul(LABEL_MULTIPLIER);
    
    //gx=tf.metrics.meanAbsoluteError(yTrue,yPred);
    console.log("X"+yPred);
    console.log("y"+yTrue);
    return tf.metrics.meanAbsoluteError(yTrue,yPred);
  });








}









const model=tf.sequential();

model.add(tf.layers.conv2d({filters:64,kernelSize:3,inputShape:[300,300,3],activation:'relu'}  ))
model.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:relu }   ))
model.add(tf.layers.maxPooling2d({poolSize:2}))


model.add(tf.layers.conv2d({filters:112,kernelSize:3,activation:relu}   ))
model.add(tf.layers.conv2d({filters:112,kernelSize:3,activation:relu}   ))
model.add(tf.layers.maxPooling2d({poolSize:2}))


model.add(tf.layers.conv2d({filters:256,kernelSize:3,activation:relu}   ))
model.add(tf.layers.conv2d({filters:256,kernelSize:3,activation:relu}   ))
model.add(tf.layers.maxPooling2d({poolSize:2}))

model.add(tf.layers.conv2d({filters:512,kernelSize:3,activation:relu}   ))
model.add(tf.layers.conv2d({filters:512,kernelSize:3,activation:relu}   ))
model.add(tf.layers.maxPooling2d({poolSize:2}))


model.add(tf.layers.conv2d({filters:1024,kernelSize:3,activation:relu}   ))
model.add(tf.layers.conv2d({filters:1024,kernelSize:3,activation:relu}   )) // layer 5 vgg-16


model.add(tf.layers.conv2d({filters:1024,kernelSize:3,activation:relu}   ))
model.add(tf.layers.conv2d({filters:1024,kernelSize:1,activation:relu}   ))
//model.add(tf.layers.upSampling2d({poolSize:2}))


model.add(tf.layers.conv2d({filters:256,kernelSize:1,activation:relu}   ))
model.add(tf.layers.conv2d({filters:512,kernelSize:3,activation:relu}   ))

//model.add(tf.layers.upSampling2d({poolSize:2}))


model.add(tf.layers.conv2d({filters:128,kernelSize:1,activation:relu}   ))
model.add(tf.layers.conv2d({filters:256,kernelSize:3,activation:relu}   ))


model.add(tf.layers.conv2d({filters:128,kernelSize:1,activation:relu}   ))
model.add(tf.layers.conv2d({filters:256,kernelSize:3,activation:relu}   ))



model.add(tf.layers.conv2d({filters:128,kernelSize:1,activation:relu}   ))

model.add(tf.layers.conv2d({filters:256,kernelSize:1,activation:relu}   ))


model.add(tf.layers.conv2d({filters:(1+4)*1,kernelSize:3,activation:relu}   ))






var pred=null,decode=null;



 var  csvss= fs.readFileSync('./ans.csv',{encoding:'utf-8'});

csvss=csvss.split('\n')
csvss=csvss.splice(1,38);

var corrdinateobj=[],csvobj=null,csv=null;
var temp=[],ys=[];

for(i=0;i<csvss.length;i++){
 csvobj={name:'' , x1:0,y1:0,x2:0,y2:0};

temp=[];
    csv= csvss[i].split(',');
csvobj.name=csv[0];
csvobj.x1=csv[4];
csvobj.y1=csv[5];
csvobj.x2=csv[6];
csvobj.y2=csv[7];
corrdinateobj.push(csvobj);
temp.push(Math.random())
temp.push(csv[4])
temp.push(csv[5])
temp.push(csv[6])
temp.push(csv[7])

ys.push(temp);
}



console.log(corrdinateobj)


var x=[];

for (let i = 0; i < 20; i++) {

    var buffer=fs.readFileSync('./dataset/'+ corrdinateobj[i].name);
     if( corrdinateobj[i].name.substr(corrdinateobj[i].name.length-3,3)=='png' ){

        continue;
     }
     
    
     


    t=tf.node.decodeJpeg(buffer)

    console.log(t.shape);
    t=tf.image.resizeNearestNeighbor(t,[300,300])
    //t=t.expandDims(0)
   // t=t.reshape([1,300,300,3])
    x.push(t)





  
 }

// var t=tf.data.Dataset({x:x,y:corrdinateobj})

 model.compile({optimizer:tf.train.adam(0.001),loss:loss,metrics:['accuracy']})

//model.fitDataset(t,{batchesPerEpoch:4})
//x1: '1559', y1: '861', x2: '2815', y2: '1824' }


const xx=tf.data.array(x)


const yy=tf.data.array(ys);

const dataset=tf.data.zip({xs:xx,ys:yy}).batch(4)
.shuffle(4);




model.fitDataset(dataset,{
    epochs: 10,
    callbacks: {onEpochEnd: (epoch, logs) => console.log(logs.loss)}}).then(()=>{

 console.log("TRAINING DONE \t NOW TESTING  ");
 var x_test=x[10];
    x_test=x_test.expandDims(0);
    model.predict(x_test).print();



    });


   


async function train(){
var i=0,y=[];
for(i=0;i<10;i++){
 
   y= tf.tensor([[[ys[i]]]]);
    const res=await model.fit(x[i],y);











}





}

//train().then(()=>console.log("\n\nTRAINING DONE\n")).catch(err=>console.log(err))








/*


model.fit(x[0],tf.tensor([[[[0.6,1559,861,2815,1824]]]])).then(()=>{

    model.fit(x[1],tf.tensor([[[[0.8,1998,697,2404,1087]]]])).then(()=>{
        model.fit(x[2],tf.tensor([[[[0.75,1468,617,2248,1233]]]])).then(()=>{
    
            model.fit(x[3],tf.tensor([[[[0.85,760,179,1226,756]]]])).then(()=>{

                model.predict(x[4]).print() 




            });





        });










    })
    










}).catch(err=>console.log("*$*%%*"+err))
*/
model.save('file:///home/joydeep/projects/fabrik');


 
