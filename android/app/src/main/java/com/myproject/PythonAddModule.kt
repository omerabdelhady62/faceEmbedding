package com.myproject

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class PythonAddModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "PythonAdd"

    @ReactMethod
    fun add(a: Double, b: Double, promise: Promise) {
        try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(reactApplicationContext))
            }
            val py = Python.getInstance()
            val module = py.getModule("add_numbers")
            val result = module.callAttr("add", a, b)
            val sum = result.toDouble()
            promise.resolve(sum)
        } catch (e: Exception) {
            promise.reject("PYTHON_ERROR", e.message, e)
        }
    }
}
