from OpenGL.GL import *
import glfw

class Window:
    def __init__(self, windowSize, windowName, visible=True, monitor=None):
        """

        :param windowSize: a tuple, size of the window (x,y)
        :param windowName: name of the window
        :param monitor: The selected monitor to be displayed. If None, then the primary monitor is used and not in full screen mode. If other monitors specified, full screen mode is used.
        """
        self.windowSize = windowSize
        self.visible = visible
        self.monitor = monitor
        self.window = self.__setUpWindow(self.windowSize, windowName, self.monitor)
    def __setUpWindow(self,windowSize,name, monitor):
        # -------- setting window

        self.init_glfw()
        window = glfw.create_window(windowSize[0], windowSize[1], name, monitor, None)
        if not window:
            glfw.terminate()
            return
        glfw.make_context_current(window)

        return window

    @staticmethod
    def getMonitors():
        """
         Discription: get the current connected monitors
        :return: an array of monitor pointers
        """
        init_glfw()
        return glfw.get_monitors()

    def processInput(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) is glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        glfw.poll_events()


    def clearWindow(self, color, alpha=1):
        glClearColor(color[0], color[1], color[2], alpha)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    def updateWindow(self):
        # swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfw.swap_buffers(self.window)

    def init_glfw(self):
        if not glfw.init():
            print('Failed to initialize GLFW')
            return

        # configuring glfw

        glfw.window_hint(glfw.VISIBLE, self.visible)
        #glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
        #glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        #glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        #glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # enable MSAA with 4 sub-samples
        glfw.window_hint(glfw.SAMPLES,4)
