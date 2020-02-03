const _apply = v => (typeof v === 'function') ? v() : v;

// Zeros-(or whatevos-)aray with a given shape
function zerosShapedAs(shape, val = 0) {
    if (shape.length == 0) {
        return _apply(val);
    } else if (shape.length == 1) {
        return new Array(shape[0]).fill().map(() => _apply(val));
    } else {
        return new Array(shape[0])
            .fill()
            .map(() => new Array(shape[1])
                .fill()
                .map(() => _apply(val)));
    }
}

// Matrix transposition
function transpose(xs) {
    return xs[0].map((_, i) => xs.map(r => r[i]));
}

// Shape of the argument
// Some concrete examples:
//   - shape of 0 is []
//   - shape of [1, 2, 3] is [3]
//   - shape of [[1, 2, 3]] is [1, 3]
//   - shape of [[1], [2], [3]] is [3, 1]
function shape(x) {
    if (Array.isArray(x)) {
        if (Array.isArray(x[0])) {
            return [x.length].concat(shape(x[0]));
        } else {
            // 1D vector
            return [x.length];
        }
    } else {
        // 0D scalar
        return [];
    }
}

// scalar * vector
function _scvecMul(s, xs) {
    return xs.map(x => x * s);
}

function _vecMul(xs, ys) {
    const dX = shape(xs);
    const dY = shape(ys);

    if (dX.length == 0) {
        return _scvecMul(xs, ys);
    } else if (dY.length == 0) {
        return _scvecMul(ys, xs);
    } else {
        if (dX[0] !== dY[0]) {
            throw new TypeError(`Vector shapes are not aligned: ${dX} and ${dY}`);
        } else {
            return xs
                .map((x, i) => x * ys[i])
                .reduce((acc, x) => acc + x);
        }
    }
}

function _scmatMul(s, xs) {
    return xs.map(r => r.map(x => x*s));
}

function _matMul(xm, ym) {
    const dX = shape(xm);
    const dY = shape(ym); 

    if (dX.length == 0) {
        return _scmatMul(xm, ym);
    } else if (dY.length == 0) {
        return _scmatMul(ym, xm);
    } else if (dX.length == 2 && dY.length == 2 && dX[1] == dY[0]) {
        let ymT = transpose(ym);
        let z = xm.map(xRow => ymT.map(yCol => _vecMul(xRow, yCol)));
        return z;
    } else {
        throw new TypeError(`Either shapes — ${dX} and ${dY} — are not aligned, or matrices are of higher dimensions.`);
    }
}

// Dot product for array-like elements
// If `x` and `y` are scalars, then just a multiplication,
// if `x` and `y` are 1D-vectors (or one is scalar and the other is 1D),
// then performs pairwise (or scalar x vector) multiplication.
// x * y is matrix multiplication for matrices, or scalar x matrix if
// one is matrix and the other is scalar
function dot(x, y) {
    const dX = shape(x);
    const dY = shape(y);

    if (dX.length == 0 && dY.length == 0) {
        return x * y;
    } else if (dX.length == 1 || dY.length == 1) {
        return _vecMul(x, y);
    } else {
        return _matMul(x, y);
    }
}


// Pairwise operation on array-like arguments
function _op(xm, ym, op = null) {
    let dX = shape(xm);
    let dY = shape(ym);

    if (dX.length == 0 && dY.length == 0) {
        return op(xm, ym);
    } else if (dX.length == 2 && dY.length == 1 && dY[0] == dX[1]) {
        return xm.map(r => r.map((a, i) => op(a, ym[i])));
    } else if (dX.length == 1 && dY.length == 1) {
        return spread(xm, (x, i) => op(x, ym[i]));
    } else if (dX.length == dY.length) {
        return spread(xm, (x, i, j) => op(x, ym[i][j]));
    } else {
        throw new TypeError(`Unimplemented for shapes ${dX} and ${dY}`);
    }
}
const add = (xm, ym) => _op(xm, ym, (x, y) => x + y);
const mul = (xm, ym) => _op(xm, ym, (x, y) => x * y);

// Sum of all matrix/array elements divided by their count.
function mean(xs) {
    let flatXs = xs.flat();
    return flatXs.reduce((a, x) => a + x)/flatXs.length;
}

// Applies function `fn` to every element of `xs`.
function spread(xs, fn = () => { throw new Error('Not implemented') }) {
    let dX = shape(xs);
    if (dX.length == 0) {
        return fn(x);
    } else if (dX.length == 1) {
        return xs.map((x, i) => fn(x, i));
    } else {
        return xs.map((r, i) => r.map((x, j) => fn(x, i, j)));
    }
}

function std(xm) {
    let data = xm.flat(Infinity);
    let m = mean(data);
    let n = data.length;

    return Math.sqrt(
        data.reduce((sq, x) => sq + (x - m)**2)/(n - 1)
    );
}

function resample(n, ...sources) {
    let [dR, dC] = shape(sources);
    let sampler = Array(dC)
        .fill(0)
        .map((_, i) => i)
        .sort(() => Math.random() - 0.5)
        .slice(0, n);

    return sources.map(src => sampler.map((i) => src[i]));
}

export default {
    add,
    mul,
    dot,
    mean,
    resample,
    shape,
    spread,
    std,
    transpose,
    zerosShapedAs,
};