import {
    Input,
    Linear,
    MSE,
    Sigmoid,
    toposort,
    feed,
    forward_pass,
    backward_pass,
} from '../ann';

describe('Random bits', () => {
    it('performs Linear fwd pass', () => {
        let [ins, weights, bias] = [
            new Input('Ins'),
            new Input('Weights'),
            new Input('Bias')
        ];

        let linear = new Linear('Linear', ins, weights, bias);

        let nn = toposort([ins, weights, bias, linear])
        feed(nn, {
            'Ins': [6, 14, 3],
            'Weights': [0.5, 0.25, 1.4],
            'Bias': 2
        });

        forward_pass(nn);
    
        expect(linear.value).toEqual(12.7);
    });

    it('performs slightly more complex Linear fwd pass', () => {
        let [ins, weights, bias] = [
            new Input('Ins'),
            new Input('Weights'),
            new Input('Bias')
        ];

        let linear = new Linear('Linear', ins, weights, bias);

        let nn = toposort([ins, weights, bias, linear])
        feed(nn, {
            'Ins': [[-1.0, -2.0], [-1.0, -2.0]],
            'Weights': [[2.0, -3.0], [2.0, -3.0]],
            'Bias': [-3., -5]
        });

        forward_pass(nn);
        
        expect(linear.value).toEqual([[-9, 4], [-9, 4]]);
    });

    it('performs Linear + Sigmoid fwd pass', () => {
        let [ins, weights, bias] = [
            new Input('Ins'),
            new Input('Weights'),
            new Input('Bias')
        ];

        let linear = new Linear('Linear', ins, weights, bias);
        let sigmoid = new Sigmoid('Sigmoid', linear);

        let nn = toposort([ins, weights, bias, linear, sigmoid])
        feed(nn, {
            'Ins': [[-1.0, -2.0], [-1.0, -2.0]],
            'Weights': [[2.0, -3.0], [2.0, -3.0]],
            'Bias': [-3., -5]
        });

        forward_pass(nn);
        
        expect(sigmoid.value).toMatchSnapshot();
    });

    it('does the backprop, too', () => {
        let [ins, weights, bias, training] = [
            new Input('Ins'),
            new Input('Weights'),
            new Input('Bias'),
            new Input('Training')
        ];

        let linear = new Linear('Linear', ins, weights, bias);
        let activation = new Sigmoid('Sigmoid', linear);
        let cost = new MSE('Cost', training, activation);

        let nn = toposort([
            ins, weights, bias,
            linear, training, activation, cost
        ]);
        feed(nn, {
            'Ins': [[-1.0, -2.0], [-1.0, -2.0]],
            'Weights': [[2.0], [3.0]],
            'Bias': [-3.],
            'Training': [1, 2]
        });

        forward_pass(nn);
        backward_pass(nn);

        expect(nn).toMatchSnapshot();
    });
});