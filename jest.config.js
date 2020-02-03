const {defaults} = require('jest-config');
module.exports = {
  transform: {
    '^.+\\.mjs?$': 'babel-jest',
  },
  moduleFileExtensions: [...defaults.moduleFileExtensions, 'mjs'],
  testMatch: [
    '<rootDir>/(test/**/*.test.(js|jsx|ts|tsx|mjs))'
  ],
};
