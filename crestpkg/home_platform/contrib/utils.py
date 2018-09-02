# Copyright (c) 2017, IGLU consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the NECOTIS research group nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.

import sys
import subprocess
import glob
import os
import numpy as np

if (sys.version_info > (3, 0)):
	import builtins
else:
	import __builtin__ as  builtins

from panda3d.core import ClockObject, AmbientLight, VBase4, PointLight, AntialiasAttrib, TextNode, LVector3f, GraphicsOutput

from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import *
from pandac.PandaModules import *

# from keras.models import Sequential
# from keras.layers import Dense

def mat4ToNumpyArray(mat):
	return np.array([[mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
					 [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
					 [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
					 [mat[3][0], mat[3][1], mat[3][2], mat[3][3]]])


def vec3ToNumpyArray(vec):
	return np.array([vec.x, vec.y, vec.z])

class Controller(ShowBase):
	def __init__(self, scene, size=(960, 720), zNear=0.1, zFar=1000.0, fov=40.0, shadowing=False, showPosition=False,
				 cameraTransform=None, cameraMask=None):

		ShowBase.__init__(self)
		
		self.__dict__.update(scene=scene, size=size, fov=fov,
							 zNear=zNear, zFar=zFar, shadowing=shadowing, showPosition=showPosition,
							 cameraTransform=cameraTransform, cameraMask=cameraMask)

		# Find agent and reparent camera to it
		self.agent = self.scene.scene.find('**/agents/agent*/+BulletRigidBodyNode')
		self.camera.reparentTo(self.agent)
		if self.cameraTransform is not None:
			self.camera.setTransform(cameraTransform)

		if cameraMask is not None:
			self.cam.node().setCameraMask(self.cameraMask)
		lens = self.cam.node().getLens()
		lens.setFov(self.fov)
		lens.setNear(self.zNear)
		lens.setFar(self.zFar)

		# Change window size
		wp = WindowProperties()
		wp.setSize(size[0], size[1])
		wp.setTitle("Controller")
		wp.setCursorHidden(True)
		self.win.requestProperties(wp)

		self.disableMouse()

		self.time = 0
		self.centX = wp.getXSize() / 2
		self.centY = wp.getYSize() / 2
		self.win.movePointer(0, int(self.centX), int(self.centY))

		# key controls
		self.forward = False
		self.backward = False
		self.fast = 2.0
		self.left = False
		self.right = False

		# sensitivity settings
		self.movSens = 2
		self.movSensFast = self.movSens * 5
		self.sensX = self.sensY = 0.2

		# Reparent the scene to render.
		self.scene.scene.reparentTo(self.render)

		self.render.setAntialias(AntialiasAttrib.MAuto)

		self.showHuman()

		# Task
		self.globalClock = ClockObject.getGlobalClock()
		self.taskMgr.add(self.update, 'controller-update')

		self._addDefaultLighting()
		self._setupEvents()

	def _addDefaultLighting(self):
		alight = AmbientLight('alight')
		alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
		alnp = self.render.attachNewNode(alight)
		self.render.setLight(alnp)

		# NOTE: Point light following the camera
		plight = PointLight('plight')
		plight.setColor(VBase4(0.4, 0.4, 0.4, 1))
		plnp = self.cam.attachNewNode(plight)
		self.render.setLight(plnp)

		if self.shadowing:
			# Use a 512x512 resolution shadow map
			plight.setShadowCaster(True, 512, 512)

			# Enable the shader generator for the receiving nodes
			self.render.setShaderAuto()
			self.render.setAntialias(AntialiasAttrib.MAuto)

	def _setupEvents(self):

		self.escapeEventText = OnscreenText(text="ESC: Quit",
											style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.95),
											align=TextNode.ALeft, scale=.05)

		if self.showPosition:
			self.positionText = OnscreenText(text="Position: ",
											 style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.85),
											 align=TextNode.ALeft, scale=.05)

			self.orientationText = OnscreenText(text="Orientation: ",
												style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.80),
												align=TextNode.ALeft, scale=.05)

		# Set up the key input
		self.accept('escape', sys.exit)
		self.accept("w", setattr, [self, "forward", True])
		self.accept("w-up", setattr, [self, "forward", False])
		self.accept("s", setattr, [self, "backward", True])
		self.accept("s-up", setattr, [self, "backward", False])
		self.accept("a", setattr, [self, "left", True])
		self.accept("a-up", setattr, [self, "left", False])
		self.accept("d", setattr, [self, "right", True])
		self.accept("d-up", setattr, [self, "right", False])
		self.accept("shift", setattr, [self, "fast", 10.0])
		self.accept("shift-up", setattr, [self, "fast", 1.0])

	def update(self, task):

		# dt = self.globalClock.getDt()
		dt = task.time - self.time

		# handle mouse look
		md = self.win.getPointer(0)
		x = md.getX()
		y = md.getY()

		if self.win.movePointer(0, int(self.centX), int(self.centY)):
			self.agent.setH(self.agent, self.agent.getH(self.agent) - (x - self.centX) * self.sensX)
			self.agent.setP(self.agent, self.agent.getP(self.agent) - (y - self.centY) * self.sensY)
			self.agent.setR(0.0)

		linearVelocityX = 0.0
		linearVelocityY = 0.0

		if self.forward == True:
			linearVelocityY += self.movSens * self.fast
		if self.backward == True:
			linearVelocityY -= self.movSens * self.fast
		if self.left == True:
			linearVelocityX -= self.movSens * self.fast
		if self.right == True:
			linearVelocityX += self.movSens * self.fast

		linearVelocity = LVector3f(linearVelocityX, linearVelocityY, 0.0)

		# Apply the local transform to the velocity
		# XXX: use BulletCharacterControllerNode class, which already handles local transform?
		rotMat = self.agent.node().getTransform().getMat().getUpper3()
		linearVelocity = rotMat.xformVec(linearVelocity)
		linearVelocity.z = 0.0
		self.agent.node().setLinearVelocity(linearVelocity)

		if self.showPosition:
			position = self.agent.getNetTransform().getPos()
			hpr = self.agent.getNetTransform().getHpr()
			self.positionText.setText(
				'Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
			self.orientationText.setText('Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))

		self.time = task.time

		# Simulate physics
		if 'physics' in self.scene.worlds:
			self.scene.worlds['physics'].step(dt)

		# Rendering
		if 'render' in self.scene.worlds:
			self.scene.worlds['render'].step(dt)

		# Simulate acoustics
		if 'acoustics' in self.scene.worlds:
			self.scene.worlds['acoustics'].step(dt)

		return task.cont

	def step(self):
		self.taskMgr.step()

	def destroy(self):
		self.taskMgr.remove('controller-update')
		ShowBase.destroy(self)
		# this should only be destroyed by the Python garbage collector
		# StaticShowBase.instance.destroy()

#------------------edit-------------------------

	def showHuman(self):
		self.pandaActor = Actor(
			'/home/teerawat/Documents/Work/Test PView/newMHuman/mHuman7.egg', 
			{'mHuman7-MhumA_test.egg': '/home/teerawat/Documents/Work/Test PView/newMHuman/mHuman7-MhumA_test.egg'})
		self.pandaActor.setScale(0.085, 0.085, 0.085)
		self.pandaActor.setPos(42, -39, 0.03)

		self.pandaActor.reparentTo(self.scene.scene)

		#print(str("test"))
		# Loop its animation.
		self.pandaActor.loop("mHuman7-MhumA_test.egg")
		self.pandaActor.pprint()

#------------end edit--------------------------

class Viewer(ShowBase):
	def __init__(self, scene, size=(960, 720), zNear=0.1, zFar=1000.0, fov=40.0, shadowing=False, interactive=True,
				 showPosition=False, cameraMask=None):

		ShowBase.__init__(self)

		self.__dict__.update(scene=scene, size=size, fov=fov, shadowing=shadowing,
							 zNear=zNear, zFar=zFar, interactive=interactive, showPosition=showPosition,
							 cameraMask=cameraMask)

		if cameraMask is not None:
			self.cam.node().setCameraMask(self.cameraMask)
		lens = self.cam.node().getLens()
		lens.setFov(self.fov)
		lens.setNear(self.zNear)
		lens.setFar(self.zFar)

		# Change window size
		wp = WindowProperties()
		wp.setSize(size[0], size[1])
		wp.setTitle("Viewer")
		wp.setCursorHidden(True)
		self.win.requestProperties(wp)

		self.disableMouse()

		self.time = 0
		self.centX = self.win.getProperties().getXSize() / 2
		self.centY = self.win.getProperties().getYSize() / 2

		# key controls
		self.forward = False
		self.backward = False
		self.fast = 1.0
		self.left = False
		self.right = False
		self.up = False
		self.down = False
		self.human_visibility = False

		# sensitivity settings
		self.movSens = 2
		self.movSensFast = self.movSens * 5
		self.sensX = self.sensY = 0.2

		self.cam.setP(self.cam, 0)
		self.cam.setR(0)

		# reset mouse to start position:
		self.win.movePointer(0, int(self.centX), int(self.centY))

		# Reparent the scene to render.
		self.scene.scene.reparentTo(self.render)

		# Task
		self.globalClock = ClockObject.getGlobalClock()
		self.taskMgr.add(self.update, 'viewer-update')

		self._addDefaultLighting()
		self._setupEvents()

		self.drawHuman()

	def _setupEvents(self):

		self.escapeEventText = OnscreenText(text="ESC: Quit",
											style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.95),
											align=TextNode.ALeft, scale=.05)

		if self.showPosition:
			self.positionText = OnscreenText(text="Position: ",
											 style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.85),
											 align=TextNode.ALeft, scale=.05)

			self.orientationText = OnscreenText(text="Orientation: ",
												style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.80),
												align=TextNode.ALeft, scale=.05)

		# Set up the key input
		self.accept('escape', sys.exit)
		self.accept("w", setattr, [self, "forward", True])
		self.accept("shift-w", setattr, [self, "forward", True])
		self.accept("w-up", setattr, [self, "forward", False])
		self.accept("s", setattr, [self, "backward", True])
		self.accept("shift-s", setattr, [self, "backward", True])
		self.accept("s-up", setattr, [self, "backward", False])
		self.accept("a", setattr, [self, "left", True])
		self.accept("shift-a", setattr, [self, "left", True])
		self.accept("a-up", setattr, [self, "left", False])
		self.accept("d", setattr, [self, "right", True])
		self.accept("shift-d", setattr, [self, "right", True])
		self.accept("d-up", setattr, [self, "right", False])
		self.accept("r", setattr, [self, "up", True])
		self.accept("shift-r", setattr, [self, "up", True])
		self.accept("r-up", setattr, [self, "up", False])
		self.accept("f", setattr, [self, "down", True])
		self.accept("shift-f", setattr, [self, "down", True])
		self.accept("f-up", setattr, [self, "down", False])
		self.accept("shift", setattr, [self, "fast", 10.0])
		self.accept("shift-up", setattr, [self, "fast", 1.0])
		self.accept("h", setattr, [self, "human_visibility", not self.human_visibility])

	def _addDefaultLighting(self):
		alight = AmbientLight('alight')
		alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
		alnp = self.render.attachNewNode(alight)
		self.render.setLight(alnp)

		# NOTE: Point light following the camera
		plight = PointLight('plight')
		plight.setColor(VBase4(0.9, 0.9, 0.9, 1))
		plnp = self.cam.attachNewNode(plight)
		self.render.setLight(plnp)

		if self.shadowing:
			# Use a 512x512 resolution shadow map
			plight.setShadowCaster(True, 512, 512)

			# Enable the shader generator for the receiving nodes
			self.render.setShaderAuto()
			self.render.setAntialias(AntialiasAttrib.MAuto)

	def update(self, task):

		# dt = self.globalClock.getDt()
		dt = task.time - self.time

		if self.interactive:
			# handle mouse look
			md = self.win.getPointer(0)
			x = md.getX()
			y = md.getY()

			if self.win.movePointer(0, int(self.centX), int(self.centY)):
				self.cam.setH(self.cam, self.cam.getH(self.cam)
							  - (x - self.centX) * self.sensX)
				self.cam.setP(self.cam, self.cam.getP(self.cam)
							  - (y - self.centY) * self.sensY)
				self.cam.setR(0)

			# handle keys:
			if self.forward == True:
				self.cam.setY(self.cam, self.cam.getY(self.cam)
							  + self.movSens * self.fast * dt)
			if self.backward == True:
				self.cam.setY(self.cam, self.cam.getY(self.cam)
							  - self.movSens * self.fast * dt)
			if self.left == True:
				self.cam.setX(self.cam, self.cam.getX(self.cam)
							  - self.movSens * self.fast * dt)
			if self.right == True:
				self.cam.setX(self.cam, self.cam.getX(self.cam)
							  + self.movSens * self.fast * dt)
			if self.up == True:
				self.cam.setZ(self.cam, self.cam.getZ(self.cam)
							  + self.movSens * self.fast * dt)
			if self.down == True:
				self.cam.setZ(self.cam, self.cam.getZ(self.cam)
							  - self.movSens * self.fast * dt)


		if self.showPosition:
			position = self.cam.getNetTransform().getPos()
			hpr = self.cam.getNetTransform().getHpr()
			self.positionText.setText(
				'Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
			self.orientationText.setText('Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))

		if self.human_visibility:
			self.pActor.show()
		else:
			self.pActor.hide()

		self.time = task.time

		# Simulate physics
		if 'physics' in self.scene.worlds:
			self.scene.worlds['physics'].step(dt)

		# Rendering
		if 'render' in self.scene.worlds:
			self.scene.worlds['render'].step(dt)

		# Simulate acoustics
		if 'acoustics' in self.scene.worlds:
			self.scene.worlds['acoustics'].step(dt)

		return task.cont

	def capture_video(self, duration=400, fps=24, **kwargs):
		self.movie(duration=duration, fps=fps, **kwargs)

	def step(self):
		self.taskMgr.step()

	def destroy(self):
		self.taskMgr.remove('viewer-update')
		ShowBase.destroy(self)
		# this should only be destroyed by the Python garbage collector
		# StaticShowBase.instance.destroy()

#------------------edit-------------------------

	def drawHuman(self):
		self.pActor = Actor(
			"/Users/hikoyu/src/crest-home/human-models/Human.egg", 
			{'PickUpBox': '/Users/hikoyu/src/crest-home/human-models/Human-PickUpBox.egg'})
		self.pActor.setScale(0.085, 0.085, 0.085)
		#self.pActor.setPos(41, -41, 0.025)
		#self.pActor.setPos(39.5, -42, 0.025)
		self.pActor.setPos(40, -40, 0.025)
		#self.pActor.setPos(40.5, -41, 0.025)
		self.pActor.reparentTo(self.scene.scene)
		self.pActor.loop("PickUpBox")#, fromFrame=500, toFrame=650)

#------------end edit--------------------------

class ControllerPepper(ShowBase):
	def __init__(self, scene, size=(960, 720), zNear=0.1, zFar=1000.0, fov=40.0, shadowing=False, showPosition=False,
				 cameraTransform=None, cameraMask=None, take_num=None):

		ShowBase.__init__(self)
		
		self.__dict__.update(scene=scene, size=size, fov=fov,
							 zNear=zNear, zFar=zFar, shadowing=shadowing, showPosition=showPosition,
							 cameraTransform=cameraTransform, cameraMask=cameraMask)

		# Find agent and reparent camera to it
		self.agent = self.scene.scene.find('**/agents/agent*/+BulletRigidBodyNode')
		self.camera.reparentTo(self.agent)
		if self.cameraTransform is not None:
			self.camera.setTransform(cameraTransform)

		if cameraMask is not None:
			self.cam.node().setCameraMask(self.cameraMask)
		lens = self.cam.node().getLens()
		lens.setFov(self.fov)
		lens.setNear(self.zNear)
		lens.setFar(self.zFar)

		# Change window size
		wp = WindowProperties()
		wp.setSize(size[0], size[1])
		wp.setTitle("Pepper")
		wp.setCursorHidden(True)
		self.win.requestProperties(wp)

		self.disableMouse()

		self.time = 0
		self.centX = wp.getXSize() / 2
		self.centY = wp.getYSize() / 2
		self.win.movePointer(0, int(self.centX), int(self.centY))

		# key controls
		self.forward = False
		self.backward = False
		self.fast = 1.2
		self.left = False
		self.right = False
		self.rotateXl = False
		self.rotateXr = False
		self.rotateYu = False
		self.rotateYd = False
		self.ss = False
		self.cv = False
		self.openpose = False
		self.visible_human = True

		# sensitivity settings
		self.sensTranslate = 3
		self.sensTranslateFast = self.sensTranslate * 5
		self.sensRotate = 0.005

		# Reparent the scene to render.
		self.scene.scene.reparentTo(self.render)

		self.render.setAntialias(AntialiasAttrib.MAuto)

		self.take_num = take_num

		# Task
		self.globalClock = ClockObject.getGlobalClock()
		self.taskMgr.add(self.update, 'pepper-update')

		self._addDefaultLighting()
		self._setupEvents()

		self.drawHuman()

	def _addDefaultLighting(self):
		alight = AmbientLight('alight')
		alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
		alnp = self.render.attachNewNode(alight)
		self.render.setLight(alnp)

		# NOTE: Point light following the camera
		plight = PointLight('plight')
		plight.setColor(VBase4(0.4, 0.4, 0.4, 1))
		plnp = self.cam.attachNewNode(plight)
		self.render.setLight(plnp)

		if self.shadowing:
			# Use a 512x512 resolution shadow map
			plight.setShadowCaster(True, 512, 512)

			# Enable the shader generator for the receiving nodes
			self.render.setShaderAuto()
			self.render.setAntialias(AntialiasAttrib.MAuto)

	def _setupEvents(self):
		self.escapeEventText = OnscreenText(text="ESC: Quit",
											style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.95),
											align=TextNode.ALeft, scale=.05)

		if self.showPosition:
			self.positionText = OnscreenText(text="Position: ",
											 style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.85),
											 align=TextNode.ALeft, scale=.05)

			self.orientationText = OnscreenText(text="Orientation: ",
												style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.80),
												align=TextNode.ALeft, scale=.05)

		# Set up the key input
		self.accept('escape', sys.exit)

		self.accept("w", setattr, [self, "forward", True])
		self.accept("w-up", setattr, [self, "forward", False])
		self.accept("s", setattr, [self, "backward", True])
		self.accept("s-up", setattr, [self, "backward", False])
		self.accept("a", setattr, [self, "left", True])
		self.accept("a-up", setattr, [self, "left", False])
		self.accept("d", setattr, [self, "right", True])
		self.accept("d-up", setattr, [self, "right", False])

		self.accept("q", setattr, [self, "rotateXl", True])
		self.accept("q-up", setattr, [self, "rotateXl", False])
		self.accept("e", setattr, [self, "rotateXr", True])
		self.accept("e-up", setattr, [self, "rotateXr", False])

		self.accept("r", setattr, [self, "rotateYu", True])
		self.accept("r-up", setattr, [self, "rotateYu", False])
		self.accept("f", setattr, [self, "rotateYd", True])
		self.accept("f-up", setattr, [self, "rotateYd", False])

		self.accept("c", setattr, [self, "ss", True])
		self.accept("c-up", setattr, [self, "ss", False])

		self.accept("o", setattr, [self, "openpose", True])
		self.accept("o-up", setattr, [self, "openpose", False])

		self.accept("m", setattr, [self, "cv", True])
		self.accept("m-up", setattr, [self, "cv", False])

		self.accept("h", setattr, [self, "visible_human", not self.visible_human])

	def update(self, task):

		# dt = self.globalClock.getDt()
		dt = task.time - self.time

		# handle key strokes
		if self.visible_human:
			self.pActor.show()
		else:
			self.pActor.hide()

		# rotational:
		if self.rotateXl == True:
			self.agent.setH(self.agent, self.agent.getH(self.agent) + self.centX * self.sensRotate)
		if self.rotateXr == True:
			self.agent.setH(self.agent, self.agent.getH(self.agent) - self.centX * self.sensRotate)
		if self.rotateYu == True:
			self.agent.setP(self.agent, self.agent.getP(self.agent) + 20 * self.sensRotate)
			self.agent.setR(self.agent, 0)
		if self.rotateYd == True:
			self.agent.setP(self.agent, self.agent.getP(self.agent) - 20 * self.sensRotate)
			self.agent.setR(self.agent, 0)

		# translational:
		linearVelocityX = 0.0
		linearVelocityY = 0.0
		if self.forward == True:
			linearVelocityY += self.sensTranslate * self.fast
		if self.backward == True:
			linearVelocityY -= self.sensTranslate * self.fast
		if self.left == True:
			linearVelocityX -= self.sensTranslate * self.fast
		if self.right == True:
			linearVelocityX += self.sensTranslate * self.fast
		linearVelocity = LVector3f(linearVelocityX, linearVelocityY, 0.0)

		# Apply the local transform to the velocity
		# XXX: use BulletCharacterControllerNode class, which already handles local transform?
		rotMat = self.agent.node().getTransform().getMat().getUpper3()
		linearVelocity = rotMat.xformVec(linearVelocity)
		linearVelocity.z = 0.0
		self.agent.node().setLinearVelocity(linearVelocity)

		# take screenshots:
		if self.ss == True:
			ShowBase.screenshot(self, namePrefix='SS', defaultFilename= True, source=None, imageComment="")

		# capture video:
		# if duration=2, fps=16, the total capture video: 32 png files.
		if self.cv == True:
			if self.take_num is not None:
				self.movie(namePrefix='take_{:02d}_frame'.format(self.take_num), duration=9, fps=30, format='png')
			else:
				print('take_num is not set.')
	
		# invoke OpenPose
		# if self.openpose == True:
		# 	p = subprocess.Popen(
		# 		('./build/examples/openpose/openpose.bin ' 
		# 			'--display 0 ' 
		# 			'--image_dir /home/teerawat/openpose/ ' 
		# 			'--render_pose 1 ' 
		# 			'--keypoint_scale 4 ' 
		# 			'--write_images /home/teerawat/Documents/Work/imagesOpenpose/ ' 
		# 			'--write_keypoint_json /home/teerawat/Documents/Work/keypointOpenposeJSON/'), 
		# 		shell=True, stdin=subprocess.PIPE)
		# 	p.wait()
		# 	for i in glob.glob(os.path.join('/home/teerawat/openpose/', "*.jpg")):
		# 		try:
		# 			os.chmod(i, 0o777)
		# 			os.remove(i)
		# 		except OSError:
		# 			pass
		# 	for i in glob.glob(os.path.join('/home/teerawat/openpose/', "*.png")):
		# 		try:
		# 			os.chmod(i, 0o777)
		# 			os.remove(i)
		# 		except OSError:
		# 			pass

		if self.showPosition:
			position = self.agent.getNetTransform().getPos()
			hpr = self.agent.getNetTransform().getHpr()
			self.positionText.setText(
				'Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
			self.orientationText.setText('Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))

		self.time = task.time

		# Simulate physics
		if 'physics' in self.scene.worlds:
			self.scene.worlds['physics'].step(dt)

		# Rendering
		if 'render' in self.scene.worlds:
			self.scene.worlds['render'].step(dt)

		# Simulate acoustics
		if 'acoustics' in self.scene.worlds:
			self.scene.worlds['acoustics'].step(dt)

		return task.cont

	def step(self):
		self.taskMgr.step()

	def destroy(self):
		self.taskMgr.remove('peppers-update')
		ShowBase.destroy(self)
		# this should only be destroyed by the Python garbage collector
		# StaticShowBase.instance.destroy()

#------------------edit-------------------------

	def drawHuman(self):
		# self.pActor = Actor(
		# 	"/home/teerawat/Documents/Human Motions/Activities/Human.egg", 
		# 	{'Crouch': '/home/teerawat/Documents/Human Motions/Activities/Human-Crouch.egg'})
		# self.pActor.setScale(0.085, 0.085, 0.085)
		# self.pActor.setPos(41, -41, 0.025)
		# self.pActor.setPos(39.5, -42, 0.025)
		# self.pActor.setPos(40, -40, 0.025)
		# self.pActor.setHpr(90, 0, 0)
		# self.pActor.setPos(40.5, -41, 0.025)
		# self.pActor.reparentTo(self.scene.scene)
		# self.pActor.loop("look")#, fromFrame=500, toFrame=650)

#------Play Multi Animation at the same time.
		# self.pActor = Actor(
		#   "/home/teerawat/Documents/Human Motions/Activities/Human.egg", 
		#   {'Crouch': '/home/teerawat/Documents/Human Motions/Activities/Human-Crouch.egg', 
		#    'Walk': '/home/teerawat/Documents/Human Motions/Activities/Human-Walk.egg'})
		# self.pActor.setScale(0.085, 0.085, 0.085)
		# self.pActor.setPos(40, -40, 0.025)
		# self.pActor.reparentTo(self.scene.scene)
		# self.pActor.enableBlend()
		# self.pActor.setControlEffect('Crouch', 0.2)
		# self.pActor.setControlEffect('Walk', 0.8)
		# self.pActor.loop('Crouch')
		# self.pActor.loop('Walk')
#------End Play Multi Animation at the same time.
		self.pActor = Actor(
			"/Users/hikoyu/src/crest-home/human-models/Human.egg", 
			{'Move': '/Users/hikoyu/src/crest-home/human-models/Human-Crouch.egg'})
		self.pActor.setScale(0.085, 0.085, 0.085)
		self.pActor.setPos(43.22, -36.80, 0.025)
		self.pActor.setHpr(-126.4, 0., 0.)
		self.pActor.reparentTo(self.scene.scene)

		self.actionSequence = Sequence(
			self.pActor.actorInterval('Move'), 
			name = 'crouchSequence')
		self.actionSequence.loop()
#------------end edit--------------------------
