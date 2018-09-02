#!/usr/bin/env python

from direct.showbase.ShowBase import ShowBase, WindowProperties
import numpy as np
import panda3d.core as pc
import snippets, sys, colorsys, random, time, os, argparse


def bigRedButton(obj):
	obj.graphicsEngine.removeAllWindows()
	obj.destroy()
	sys.exit()


def assignTask(obj, task, name):
	if not obj.taskMgr.hasTaskNamed(name):
		obj.taskMgr.add(task, name)

class StaticSpace(ShowBase):
	def __init__(self, mode='data', winsize=(512, 512), ntotalscene=20, nperscene=5, fromscene=0):
		super().__init__()
		
		self.setBackgroundColor(.5294, .8078, .9804)

		wp = WindowProperties()
		wp.setSize(*winsize)
		self.win.requestProperties(wp)
		
		self.manifest_egg = {
			'floor':  ['models/space_static/egg/floortex00.egg',
					   'models/space_static/egg/floortex01.egg',
					   'models/space_static/egg/floortex02.egg'],
			'wall':   ['models/space_static/egg/walltex00.egg',
					   'models/space_static/egg/walltex01.egg',
					   'models/space_static/egg/walltex02.egg',
					   'models/space_static/egg/walltex03.egg',
					   'models/space_static/egg/walltex04.egg'],
			'object': ['models/space_static/egg/obj00.egg',
					   'models/space_static/egg/obj01.egg',
					   'models/space_static/egg/obj02.egg',
					   'models/space_static/egg/obj03.egg',
					   'models/space_static/egg/obj04.egg',
					   'models/space_static/egg/obj05.egg',
					   'models/space_static/egg/obj06.egg']
		}
		
		self.manifest_model = {
			'floor':  [self.loader.loadModel(egg) for egg in self.manifest_egg['floor']],
			'wall':   [self.loader.loadModel(egg) for egg in self.manifest_egg['wall']],
			'object': [self.loader.loadModel(egg) for egg in self.manifest_egg['object']]
		}
		
		self.space_object_position = np.mgrid[-2:2:.01, -2:2:.01].reshape([2, -1]).T.tolist()
		self.space_camera_position = np.mgrid[-3:3:.05, -3:3:.05, .1:1.5:.01].reshape([3, -1]).T.tolist()
		self.space_camera_hpfacing = np.mgrid[0:359:1, -50:8:1].reshape([2, -1]).T.tolist()
		self.space_light_position = np.mgrid[-4:4:.3, -3:3:.3].reshape([2, -1]).T.tolist()
		
		self.mainSceneNp = self.render.attachNewNode('mainSceneNodePath')
		self.objParentNp = self.render.attachNewNode('objectParentNodePath')
		
		self.setup_default_lights()
		self.setup_default_camera()
		
		self.draw_random_scene()
		self.draw_random_object()
		
		self.accept('escape', bigRedButton, [self])
		if mode == 'view':
			self.setup_default_event_handler()
			self.accept('i', assignTask, [self, self.task_view, 'taskView'])
		elif mode == 'data':
			self.xyzhpList = []
			self.sceneCount = fromscene
			self.viewPerScene = nperscene
			self.viewCount = 0
			self.totalSceneCount = ntotalscene - self.sceneCount
			self.accept('i', assignTask, [self, self.task_data, 'taskData'])

		print(snippets.listNodePath(self.render))
		print("*** Press `i' to initiate `{}' task. ***".format(mode))
		
	def setup_default_event_handler(self):
		self.eventFlag = {
			'scene': False,
			'object': False,
			'light': False,
			'camera': False
		}
		
		self.eventTimeStamp = {
			'scene': 0,
			'object': 0,
			'camera': 0,
			'light': 0
		}
		
		def registerKey(key, value):
			if type(value) is bool:
				self.eventFlag[key] = value
			else:
				if value - self.eventTimeStamp[key] > 0.05:
					self.eventTimeStamp[key] = value
					self.eventFlag[key] = True

		self.accept('time-s', registerKey, ['scene'])
		self.accept('time-o', registerKey, ['object'])
		self.accept('time-c', registerKey, ['camera'])
		self.accept('time-l', registerKey, ['light'])
		self.accept('s-up', registerKey, ['scene', False])
		self.accept('o-up', registerKey, ['object', False])
		self.accept('c-up', registerKey, ['camer', False])
		self.accept('l-up', registerKey, ['light', False])
		
	def setup_default_lights(self, frustum=False):
		self.lightNpSpot = self.render.attachNewNode(pc.Spotlight('lightSpot'))
		self.lightNpSpot.node().setColor(pc.VBase4(.8, .8, .8, 1.))
		self.lightNpSpot.node().setLens(pc.PerspectiveLens())
		self.lightNpSpot.node().setShadowCaster(True, 2048, 2048)
		self.lightNpSpot.node().getLens().setFov(60)
		self.lightNpSpot.node().getLens().setNearFar(.1, 40)
		self.lightNpSpot.node().getLens().setFilmSize(2048, 2048)
		self.lightNpSpot.setPos(-3.5, -3.5, 8)
		self.lightNpSpot.lookAt(0, 0, 0)
		self.render.setLight(self.lightNpSpot)
		self.render.setShaderAuto()
		
		if frustum:
			self.lightNpSpot.node().showFrustum()
		
		self.lightNpAmbient = self.render.attachNewNode(pc.AmbientLight('lightAmbient'))
		self.lightNpAmbient.node().setColor(pc.VBase4(.5, .5, .5, 1.))
		self.lightNpAmbient.setPos(0, 0, 0)
		self.render.setLight(self.lightNpAmbient)
		
	def setup_default_camera(self):
		self.point_camera_random()
		# self.currentCameraPosition = [0., 0., .1]
		# self.currentCameraHPFacing = [0., 5.]
		self.camLens.setNearFar(.001, 20)
		self.camLens.setFov(60)
		
	def sample_hsv(self):
		hue = np.random.uniform(low=0., high=1., size=1)
		saturation = np.random.uniform(low=.75, high=1., size=1)
		value = 1.
		rgb = pc.Vec3(*colorsys.hsv_to_rgb(hue, saturation, value))
		return rgb
	
	def draw_random_scene(self):
		for child in self.mainSceneNp.getChildren():
			child.detachNode()
		
		# draw floor/wall
		floorModel = random.sample(self.manifest_model['floor'], 1)[0]
		wallModel = random.sample(self.manifest_model['wall'], 1)[0]
		floorModel.reparentTo(self.mainSceneNp)
		wallModel.reparentTo(self.mainSceneNp)
	
	def draw_random_object(self):
		for child in self.objParentNp.getChildren():
			child.detachNode()
			
		# draw objects, spin, set position and color
		n = random.randint(1, 3)
		for objModel, pos in zip(random.sample(self.manifest_model['object'], n), 
								 random.sample(self.space_object_position, n)):
			heading = random.randint(0, 359)
			rgb = self.sample_hsv()
			objModel.setH(heading)
			objModel.setPos(pos[0], pos[1], 0)
			objModel.setColor(rgb)
			objModel.reparentTo(self.objParentNp)
			
	def point_camera_random(self):
		source = random.sample(self.space_camera_position, 1)[0]
		hpfacing = random.sample(self.space_camera_hpfacing, 1)[0]
		heading, pitch = hpfacing
		self.currentCameraPosition = source
		self.currentCameraHPFacing = hpfacing
		self.camera.setPos(*source)
		self.camera.setH(heading)
		self.camera.setP(pitch)

		# force to update frame
		self.graphicsEngine.renderFrame()
		
		info = [source[0], source[1], source[2], np.sin(heading), np.cos(heading), np.sin(pitch), np.cos(pitch)]
		return info
	
	def place_light_random(self):
		pos = random.sample(self.space_light_position, 1)[0]
		self.lightNpSpot.setPos(pos[0], pos[1], 8)
		self.lightNpSpot.lookAt(0, 0, 0)
	
	def task_data(self, task):
		# record camera orientation
		caminfo = self.point_camera_random()
		self.xyzhpList.append(caminfo)

		# verify data path
		capture_name = '{:08d}_{:02d}.jpg'.format(self.sceneCount, self.viewCount)
		path_name = os.path.join(os.environ['HOME'], 'gqn/room/{:08d}'.format(self.sceneCount))
		if not os.path.exists(path_name):
			os.mkdir(path_name)

		# capture scene
		full_name = os.path.join(path_name, capture_name)
		self.screenshot(full_name, defaultFilename=False, source=self.win)
		
		self.viewCount = (self.viewCount + 1) % self.viewPerScene
		
		if self.viewCount == 0:
			np.save(os.path.join(path_name, 'xyzhp.npy'), self.xyzhpList)
			self.xyzhpList = []
			print('- {:08d} done'.format(self.sceneCount))

			self.sceneCount += 1
			self.draw_random_scene()
			self.place_light_random()
			self.draw_random_object()

		if self.sceneCount == self.totalSceneCount:
			print('--- All done.')
			bigRedButton(self)
			return task.done
		else:
			return task.cont
	
	def task_view(self, task):
		self.camera.setPos(*self.currentCameraPosition)
		self.camera.setH(self.currentCameraHPFacing[0])
		self.camera.setP(self.currentCameraHPFacing[1])
		
		if self.eventFlag['scene']:
			self.draw_random_scene()
			self.eventFlag['scene'] = False
			
		if self.eventFlag['light']:
			self.place_light_random()
			self.eventFlag['light'] = False
			
		if self.eventFlag['object']:
			self.draw_random_object()
			self.eventFlag['object'] = False
			
		if self.eventFlag['camera']:
			info = self.point_camera_random()
			self.eventFlag['camera'] = False
			print(info)
			
		return task.cont


def main(**kwargs):
	# mode='data', winsize=(64, 64), ntotalscene=10, nperscene=5, fromscene=0
	sspace = StaticSpace(**kwargs)
	sspace.run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, choices=['view', 'data'], dest='mode')
	parser.add_argument('--size', nargs=2, type=int, dest='winsize', default=[64, 64])
	parser.add_argument('--num', nargs=1, type=int, dest='ntotalscene', default=10)
	parser.add_argument('--per', nargs=1, type=int, dest='nperscene', default=5)
	parser.add_argument('--from', nargs=1, type=int, dest='fromscene', default=0)
	argdict = vars(parser.parse_args(sys.argv[1:]))
	print(argdict)
	main(**argdict)