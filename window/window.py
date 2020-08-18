from OpenGL.GL import *
import glfw


class Window:
    def __init__(self,
                 window_size,
                 window_name,
                 visible=True,
                 monitor=None):
        """
        Window class wrapped with glfw window
        :param window_size: a tuple, size of the window (x,y)
        :param window_name: name of the window
        :param monitor: The selected monitor to be displayed. If None, then the primary monitor is used and not in
        full screen mode. If other monitors specified, full screen mode is used.
        """
        self.window_size = window_size
        self.visible = visible
        self.monitor = monitor
        self.window = self.__set_up_window(self.window_size, window_name, self.monitor)

    def __set_up_window(self, window_size, name, monitor):
        """
        Creating window object using pyglfw api "create_window"
        :param window_size: tuple, (width, height)
        :param name: str, title of the window
        :param monitor: glfw monitor
        :return: glfw window object
        """

        self.init_glfw()
        window = glfw.create_window(window_size[0], window_size[1], name, monitor, None)
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

    def process_input(self):
        """
        Process inputs of the window, currently handles only the escape key
        :return: None 
        """
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) is glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        glfw.poll_events()

    @staticmethod
    def clear_window(color, alpha=1):
        """
        Clear window with specific values 
        :param color: RGB tuple/list/array 
        :param alpha: [0,1] alpha value
        :return: None
        """
        glClearColor(color[0], color[1], color[2], alpha)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def update_window(self):
        """
        Swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        :return:
        """
        glfw.swap_buffers(self.window)

    def init_glfw(self):
        """
        Initialize glfw with window hints
        :return: 
        """
        if not glfw.init():
            print('Failed to initialize GLFW')
            return

        # configuring glfw

        glfw.window_hint(glfw.VISIBLE, self.visible)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # enable MSAA with 4 sub-samples
        glfw.window_hint(glfw.SAMPLES, 4)
