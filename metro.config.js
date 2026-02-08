const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);
const { assetExts } = defaultConfig.resolver;

const config = {
    resolver: {
        // Add 'tflite' to the existing list of asset extensions
        assetExts: [...assetExts, 'tflite'],
    },
};

module.exports = mergeConfig(defaultConfig, config);