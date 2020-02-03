import math from './math.mjs';

export class Node {
    constructor(name, inbound = []) {
        this.name = name;

        // Receiving input from inbound
        this.inbound = inbound;

        // Sending output to outbound
        this.outbound = [];

        // Value
        this.value = null;

        // Hashmap, with keys being node names,
        // and values being partials of the node with respect
        // to the input.
        this.gradients = {};

        // Making ourselves an outbound node for all the inbound ones
        this.inbound.forEach((n) => n.outbound.push(this));
    }

    forward() {
        throw new Error('Not implemented.');
    }

    backward() {
        throw new Error('Not implemented.');
    }

    dbg(name, ...log) {
        if(name === this.name) {
            console.log(`DBG ${name}`, ...log);
        }
    }
}

export class Input extends Node {
    constructor(name) {
        super(name, []);
    }

    forward() {
        // console.debug('Forward', this.name);
        // I am an input node, I don't want to calculate anything.
    }

    backward() {
        // console.debug('Backwards', this.name);
        // An input node has no input, so the gradient (derivative)
        // is zero.
        this.gradients[this.name] = math.zerosShapedAs(math.shape(this.outbound[0].gradients[this.name]));

        this.outbound.forEach(n => {
            this.gradients[this.name] = math.add(
                this.gradients[this.name],
                n.gradients[this.name]
            );
        });
        
    }
}

export class Linear extends Node {
    constructor(name, X, W, b) {
        super(name, [X, W, b]);
    }

    forward() {
        // console.debug('Forward', this.name);
        // Do the linear transform
        let [X, W, b] = this.inbound.map(n => n.value);

        this.value = math.add(math.dot(X, W), b);
    }

    backward() {
        // console.debug('Backwards', this.name);
        this.gradients = {};
        this.inbound.forEach(n => {
            this.gradients[n.name] = math.zerosShapedAs(math.shape(n.value));
        });

        let inpts = this.gradients[this.inbound[0].name];
        let wghts = this.gradients[this.inbound[1].name];
        let biass = this.gradients[this.inbound[2].name];


        this.outbound.forEach(n => {
            let gradCost = n.gradients[this.name];
            // Inputs
            inpts = math.add(
                inpts,
                math.dot(gradCost, math.transpose(this.inbound[1].value))
            );
            // Weights
            wghts = math.add(
                wghts,
                math.dot(math.transpose(this.inbound[0].value), gradCost)
            );
            biass = math.add(biass, math.transpose(gradCost).map(c => c.reduce((a, x) => a+x)));
            //this.dbg('L1', biass);

        });

        this.gradients[this.inbound[0].name] = inpts;
        this.gradients[this.inbound[1].name] = wghts;
        this.gradients[this.inbound[2].name] = biass;
    }
}

export class Sigmoid extends Node {
    constructor(name, node) {
        super(name, [node]);
    }

    _sigmoid(x) {
        return math.spread(x, a => 1.0/(1.0 + Math.exp(-a)));
    }
    forward() {
        let input = this.inbound[0].value;
        this.value = this._sigmoid(input);
    }

    backward() {
        this.gradients = {};
        this.inbound.forEach(n => {
            this.gradients[n.name] = math.zerosShapedAs(math.shape(n.value));
        });

        let ing = this.gradients[this.inbound[0].name];
        this.outbound.forEach(n => {
            let gradCost = n.gradients[this.name];
            let sigmoid = this.value;
            ing = math.add(
                ing,
                math.mul(
                    math.spread(sigmoid, (s) => s * (1-s)),
                    gradCost
                )
            );
        });
        this.gradients[this.inbound[0].name] = ing;
    }
}

export class MSE extends Node {
    constructor(name, result, target) {
        super(name, [result, target]);

        this.diff = null;
    }

    forward() {
        // console.debug('Forward', this.name);
        let [result, target] = this.inbound.map(x => [x.value].flat(Infinity));
        this.m = target.length;

        this.diff = math.transpose([math.add(target, math.dot(-1, result))]);
        this.value = math.mean(math.spread(this.diff, x => x*x));
    }

    backward() {
        this.gradients[this.inbound[0].name] = math.dot((-2/this.m), this.diff);
        this.gradients[this.inbound[1].name] = math.dot((2/this.m), this.diff);
    }
}

// Topologically sorts directed graph, represented by `nodes`.
export function toposort(nodes) {
    // An array of nodes and their `.outbound` fields
    // automatically form an adjacency list.

    // Visitation marks for the BFS
    let marked = {};
    nodes.forEach(n => {marked[n.name] = false});

    let sorted = [];
    let bfs = (v) => {
        marked[v.name] = true;
        v.outbound.forEach(w => { if(!marked[w.name]) bfs(w) });
        sorted.push(v);
    }

    // Do sorting for each input node
    // _Input_ here means there are no inbound nodes
    nodes.forEach(n => {
        if(n.inbound.length == 0) {
            bfs(n);
        }
    });

    return sorted;
}

export function feed(xs, dict) {
    xs.forEach(x => x.value = dict[x.name]);
}

export function forwardPass(nn) {
    nn.slice().reverse().forEach(n => n.forward());
}

export function backwardPass(nn) {
    nn.forEach(n => n.backward());
}

// Stochastic Gradient Descent
export function sgdUpdate(trainables, learningRate = 0.1) {
    // Change the trainable's value by subtracting the learning rate
    // multiplied by the partial of the cost with respect to this
    // trainable.
    trainables.forEach(t => {
        let partial = t.gradients[t.name];
        t.value = math.add(
            t.value,
            math.spread(partial, x => - x*learningRate)
        );
    });

}