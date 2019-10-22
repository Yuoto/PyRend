from OpenGL.GL import *
import glfw

class Window:
    def __init__(self, windowSize, windowName):
        self.windowSize = windowSize
        self.window = self.__setUpWindow(self.windowSize, windowName)
    def __setUpWindow(self,windowSize,name):
        # -------- setting window

        init_glfw()
        window = glfw.create_window(windowSize[0], windowSize[1], name, None, None)
        if not window:
            glfw.terminate()
            return
        glfw.make_context_current(window)

        return window


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

def init_glfw():
    if not glfw.init():
        print('Failed to initialize GLFW')
        return

    # configuring glfw
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # enable MSAA with 4 sub-samples
    glfw.window_hint(glfw.SAMPLES,4)
