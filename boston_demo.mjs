import {
    Input,
    Linear,
    MSE,
    Sigmoid,
    toposort,
    feed,
    forwardPass,
    backwardPass,
    sgdUpdate,
} from './ann.mjs';

import math from './math.mjs';

import data from './boston_housing.json';

// Normalizing the data, columnwise
let Xmean = math.transpose(data.data).map(col => math.mean(col));
let Xstd = math.transpose(data.data).map(col => math.std(col));
let X_ = math.spread(data.data, (x, i, j) => (x - Xmean[j])/Xstd[j]);

// Building up the rest of the inputs
let y_ = data.target;

let numFeatures = math.shape(X_)[1];
let numHidden = 10;

let W1_ = math.zerosShapedAs([numFeatures, numHidden], () => Math.random());
let b1_ = math.zerosShapedAs([numHidden]);
let W2_ = math.zerosShapedAs([numHidden, 1], () => Math.random());
let b2_ = math.zerosShapedAs([1]);

// Defining the network
let [X, y,
    W1, b1,
    W2, b2] = [
        new Input('X'), new Input('y'),
        new Input('W1'), new Input('b1'),
        new Input('W2'), new Input('b2'),
    ];

let l1 = new Linear('L1', X, W1, b1);
let s = new Sigmoid('s', l1);
let l2 = new Linear('L2', s, W2, b2);
let cost = new MSE('MSE', y, l2);

let nn = toposort([X, y, W1, b1, W2, b2, l1, s, l2, cost]);
feed(nn, {
    'X': X_, 'y': y,
    'W1': W1_, 'b1': b1_, 'W2': W2_, 'b2': b2_
});

let trainables = [W1, b1, W2, b2];

const epochs = 1000;
const m = math.shape(X_)[0];
const batchSize = 11;
const stepsPerEpoch = Math.floor(m / batchSize);

for(let i = 0; i < epochs; i++) {
    let loss = 0;
    for(let j = 0; j < stepsPerEpoch; j++) {
        let [Xbatch, ybatch] = math.resample(batchSize, X_, y_);
        //let [Xbatch, ybatch] = [X_.slice(0,3), y_.slice(0,3)];
        X.value = Xbatch;
        y.value = ybatch;

        forwardPass(nn);
        backwardPass(nn);
        sgdUpdate(trainables, 0.001);
        //console.table(trainables.map(n => [n.name, n.value, n.gradients[n.name]]));

        loss += nn[0].value;
    }
    console.log(`Epoch ${i}, loss ${loss/stepsPerEpoch}`);
}
