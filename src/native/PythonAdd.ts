/**
 * Native module that runs the add logic:
 * - Android: Chaquopy runs add_numbers.py
 * - iOS: Swift module (same logic, no Python on device)
 */

import { NativeModules } from 'react-native';

const { PythonAdd: NativePythonAdd } = NativeModules;

export interface PythonAddSpec {
  add(a: number, b: number): Promise<number>;
}

export const PythonAdd: PythonAddSpec = {
  add(a: number, b: number): Promise<number> {
    if (NativePythonAdd == null) {
      return Promise.reject(
        new Error('PythonAdd native module is not available'),
      );
    }
    return NativePythonAdd.add(a, b);
  },
};
