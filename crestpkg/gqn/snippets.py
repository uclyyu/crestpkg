from panda3d.core import (LineStream, Loader, LoaderOptions, NodePath, Filename,
						  ClockObject, AmbientLight, VBase4, PointLight, AntialiasAttrib, TextNode, LVector3f)
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase, WindowProperties
import os, sys

def listNodePath(nodepath):
	text = ''
	line_stream = LineStream()
	nodepath.ls(line_stream)
	while line_stream.isTextAvailable():
		text += line_stream.getLine() + '\n'

	return text


def loadModel(modelPath):
	loader = Loader.getGlobalPtr()
	# NOTE: disable disk and RAM caching to avoid filling memory when loading multiple scenes
	loaderOptions = LoaderOptions(LoaderOptions.LF_no_cache)
	node = loader.loadSync(Filename(modelPath), loaderOptions)
	if node is not None:
		nodePath = NodePath(node)
		nodePath.setTag('model-filename', os.path.abspath(modelPath))
	else:
		raise IOError('Could not load model file: %s' % (modelPath))
	return nodePath