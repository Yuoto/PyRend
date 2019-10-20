# PyOpenGLRenderer

<b>Depedencies:</b>
<ul>
<li>PyOpenGL</li>
<li>pyglfw 1.8.1</li>
<li>numpy</li>
<li>scipy 1.0.1</li>
<li>pyassimp 4.1.3</li>
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

Notice1: Since scipy removed imread & imsave in higher version, a lower scipy version is needed. (Will be fixed recently)
Notice2: After installing pyassimp via pip, for windows users, you might need to put the dynamic libraries as well as binaries assimp.dll & assimp.lib (Precompiled binaries can be downloaded, however for the latest version a compilation from source code is needed) into the directory where pyassimp/helper.py lies in (usually it's at Pythonxx/Lib/site-packages/pyassimp/). As for linux users, simply install libassimp-dev using apt-get.


