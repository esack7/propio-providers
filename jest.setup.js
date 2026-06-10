// Setup file to make Jest globals available in ESM mode
import {
  jest,
  expect,
  describe,
  it,
  beforeEach,
  afterEach,
  beforeAll,
  afterAll,
} from "@jest/globals";

// Inject globals into the global scope
globalThis.jest = jest;
globalThis.expect = expect;
globalThis.describe = describe;
globalThis.it = it;
globalThis.beforeEach = beforeEach;
globalThis.afterEach = afterEach;
globalThis.beforeAll = beforeAll;
globalThis.afterAll = afterAll;
