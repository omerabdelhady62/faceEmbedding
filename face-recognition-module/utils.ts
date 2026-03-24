import { Image } from 'react-native'
import RNFS from 'react-native-fs'
import { fetchDb, deleteDb, type AppDbs, type DbName } from './db'

export { fetchDb, deleteDb, type DbName }

export function getTime(): number {
  return Date.now()
}

export async function resolveImageToLocalUri(source: any, tag: string): Promise<string> {
  let uri: string

  if (typeof source === 'number') {
    const resolved = Image.resolveAssetSource(source)
    uri = resolved.uri
  } else if (typeof source === 'string') {
    uri = source
  } else {
    throw new Error(`Unsupported image source type for ${tag}`)
  }

  const extension = uri.toLowerCase().includes('.png') ? 'png' : 'jpg'
  const destPath = `${RNFS.TemporaryDirectoryPath}/img_${tag}_${Date.now()}.${extension}`

  const res = await RNFS.downloadFile({ fromUrl: uri, toFile: destPath }).promise

  if (res.statusCode && res.statusCode >= 400) {
    throw new Error(`Failed to resolve image ${tag} (status ${res.statusCode})`)
  }

  return `file://${destPath}`
}
