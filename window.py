from OpenGL.GL import *
import glfw
import time
import numpy as np
from camera import direction

lastFrame = 0

class Window:
    def __init__(self,
                 width,
                 height,
                 name,
                 visible=True,
                 enableUI = False,
                 monitor=None):
        """
        Window class wrapped with glfw window
        :param width: width of the window
        :param height: height of the window
        :param name: name of the window
        :param monitor: The selected monitor to be displayed. If None, then the primary monitor is used and not in
        full screen mode. If other monitors specified, full screen mode is used.
        """
        self.width = width
        self.height = height

        self.visible = visible
        self.glfwMonitor = monitor
        self.glfwWindow = self.__setupWindow(name)
        self.enableUI = enableUI

    def __setupWindow(self, name):
        """
        Creating window object using pyglfw api "create_window"
        :param name: str, title of the window
        :return: glfw window object
        """

        self.initGLFW()
        # if self.glfwMonitor is None:
        #     self.glfwMonitor = glfw.get_primary_monitor()

        window = glfw.create_window(self.width, self.height, name, self.glfwMonitor, None)
        if not window:
            glfw.terminate()
            return
        glfw.make_context_current(window)

        return window

    @staticmethod
    def get_monitors():
        """
         Description: get the current connected monitors
        :return: an array of monitor pointers
        """
        if not glfw.init():
            print('Failed to initialize GLFW')
        return glfw.get_monitors()

    def processInput(self, cam, deltaT=0.01):
        global lastFrame
        currentFrameTime = glfw.get_time()
        deltaT = currentFrameTime - lastFrame
        lastFrame = currentFrameTime

        if glfw.get_key(self.glfwWindow, glfw.KEY_ESCAPE) is glfw.PRESS:
            glfw.set_window_should_close(self.glfwWindow, True)

        if self.enableUI:

            if glfw.get_key(self.glfwWindow, glfw.KEY_W) == glfw.PRESS:
                cam.processKeyBoard(direction.FORWARD, deltaT)
            if glfw.get_key(self.glfwWindow, glfw.KEY_S) == glfw.PRESS:
                cam.processKeyBoard(direction.BACKWARD, deltaT)
            if glfw.get_key(self.glfwWindow, glfw.KEY_A) == glfw.PRESS:
                cam.processKeyBoard(direction.LEFT, deltaT)
            if glfw.get_key(self.glfwWindow, glfw.KEY_D) == glfw.PRESS:
                cam.processKeyBoard(direction.RIGHT, deltaT)

    def updateWindow(self):
        """
        Swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        :return:
        """
        glfw.swap_buffers(self.glfwWindow)
        glfw.poll_events()

    def initGLFW(self):
        """
        Initialize glfw with window hints
        :return: 
        """
        if not glfw.init():
            print('Failed to initialize GLFW')
            return

        # configuring glfw

        glfw.window_hint(glfw.VISIBLE, self.visible)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # if for APPLE OS
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # enable MSAA with 4 sub-samples
        glfw.window_hint(glfw.SAMPLES, 4)
