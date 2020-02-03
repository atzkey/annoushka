import math from '../math';

describe('math', () => {
    describe('dot multiplication', () => {
        it('multiplies vector by scalar', () => {
            expect(math.dot(2, [1, 2, 3])).toEqual([2, 4, 6]);
            expect(math.dot([1, 2, 3], 3)).toEqual([3, 6, 9]);
        });

        it('multiplies vector by vector, when aligned', () => {
            expect(math.dot([1, 2, 3], [4, 5, 6])).toEqual(32);
        });

        it('fails to multiply misaligned vectors', () => {
            expect(() => math.dot([1, 2], [1])).toThrowError(TypeError);
        });

        it('multiplies matrix by scalar', () => {
            expect(math.dot(2, [[1, 2], [3, 4]])).toEqual(
                [[2, 4], [6, 8]]
            );
            expect(math.dot([[1, 2], [3, 4]], 3)).toEqual(
                [[3, 6], [9, 12]]
            );
        });

        it('multiplies two square matrices', () => {
            expect(math.dot(
                // X
                [[1, 3, 3], 
                [1, 4, 3],
                [1, 3, 4]],
                // Y
                [[7, -3, -3],
                [-1, 1, 0],
                [-1, 0, 1]]
            )).toEqual(
                [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
            );
        });

        it('multiplies two aligned matrices', () => {
            expect(math.dot(
                // X(2, 3)
                [[1, 2, 4], 
                [3, 5, 6]],
                // Y(3, 4)
                [[6, 5, 1, 2], 
                [4, 3, 3, 4],
                [2, 1, 5, 6]]
            )).toEqual(
                [[22, 15, 27, 34], // X*Y(2,4)
                [50, 36, 48, 62]]
            );
        });
    });

    describe('misc', () => {
        it('means something', () => {
            expect(math.mean([[1, 2], [3, 4]])).toEqual(2.5);
            expect(math.mean([[3, 4]])).toEqual(3.5);
            expect(math.mean([4, 5])).toEqual(4.5);
        });
    });
});