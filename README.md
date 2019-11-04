# PyOpenGLRenderer

<b>Depedencies:</b>
<ul>
<li>PyOpenGL</li>
<li>glfw 1.8.1</li>
<li>numpy</li>
<li>scipy 1.0.1</li>
<li>pyassimp 4.1.3</li>
<li>imgui 1.0.0</li>
</ul>	
<br />

<b>Steps:</b>
<ol>
<li>Setup Shader paths</li>
<li>Setup model paths</li>
<li>Setup window</li>
<li>Create Light/Camera/Renderer</li>
<li>Setup Light Attribute(& pose)/Model Attribute(& pose)</li>
<li>Draw</li>
</ol>
 
<br />
Run the script "rendererTest.py" to test renderer.<br />

<ul>
<li><b>Notice1</b>: Since scipy removed imread & imsave in higher version, a lower scipy version is needed. (Will be fixed recently)</li>
<li><b>Notice2</b>: The 4.1.4 latest version of pyassimp might crase, hence use version 4.1.3 instead! After installing pyassimp 4.1.3 via pip, for windows users, you might need to put the dynamic libraries as well as binaries assimp.dll & assimp.lib (Precompiled binaries can be downloaded, however for the latest version a compilation from source code is needed) into the directory where pyassimp/helper.py lies in (usually it's at Pythonxx/Lib/site-packages/pyassimp/). As for linux users, simply install libassimp-dev using apt-get.</li>
</ul>

