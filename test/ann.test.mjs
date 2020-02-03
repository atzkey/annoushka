import 'assert';

import {
    Node, toposort
} from '../ann';

describe('Ann', () => {
    describe('Node', () => {
        let nA = new Node('A'),
            nB = new Node('B'),
            nC = new Node('C', [nA, nB]);

        it('registers itself as outbound', () => {
            expect(nA.outbound).toEqual([nC]);
            expect(nB.outbound).toEqual([nC]);
            expect(nC.outbound).toEqual([]);
        });
    });

    describe('toposort', () => {
        /** 
        *   .─.      .─.     .─.     │  Adjacency list:
        *  ( 1 )───▶( 3 )──▶( 5 )    |  1: 3
        *   `─'      `─'     `─'     |  2: 3, 4
        *             ▲       ▲      |  3: 5
        *             │       │      |  4: 5
        *            .─.     .─.     |  5: -
        *           ( 2 )──▶( 4 )    |
        *            `─'     `─'     |
        */
    
        let v1 = new Node('1', []),
            v2 = new Node('2', []),
            v3 = new Node('3', [v1, v2]),
            v4 = new Node('4', [v2]),
            v5 = new Node('5', [v4, v3]);
        
        it('topologically sorts, eh?', () => {
            expect(toposort([v1, v2, v3, v4, v5])).toEqual(
                [v5, v3, v1, v4, v2]
            );
        });
    });
});

