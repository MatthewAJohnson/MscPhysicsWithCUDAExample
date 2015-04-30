#include "Renderer.h"

Renderer* Renderer::instance = NULL;

Renderer::Renderer(Window &parent) : OGLRenderer(parent)	{	
	camera			= NULL;

	root			= new SceneNode();
			
	simpleShader	= new Shader(SHADERDIR"TechVertex.glsl", SHADERDIR"TechFragment.glsl");
	
	basicText		= new Font(SOIL_load_OGL_texture(TEXTUREDIR"tahoma.tga",SOIL_LOAD_AUTO,SOIL_CREATE_NEW_ID,SOIL_FLAG_COMPRESS_TO_DXT),16,16);
	
	if(!simpleShader->LinkProgram() )
	{
		return;
	}
	
	
	instance		= this;

	fpsTimer		= 0;
	
	fps				= 0;

	frameCounter	= 0;

	init			= true;

	water = new VBOWaterResource();
}

Renderer::~Renderer(void)	{
	delete root;
	delete simpleShader;
	delete water;
	currentShader = NULL;
}

void Renderer::UpdateScene(float msec)	{
	if(camera) {
		camera->UpdateCamera(msec); 
	}
	root->Update(msec);
	water->update(msec);
//	collisions = CollisionHelper::collisionCounter;

	//fpsTimer += msec;
	//frameCounter++;
	//if(fpsTimer >= 1000)
	//{
	//	fps = (float)frameCounter / (fpsTimer * 0.001f);
	//	frameCounter = 0;
	//	fpsTimer = 0;
	//}


}

void Renderer::RenderScene()	{
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	if(camera) {
		SetCurrentShader(simpleShader);
		glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "diffuseTex"), 0);

		textureMatrix.ToIdentity();
		modelMatrix.ToIdentity();
		viewMatrix		= camera->BuildViewMatrix();
		projMatrix		= Matrix4::Perspective(1.0f,10000.0f,(float)width / (float) height, 45.0f);
		frameFrustum.FromMatrix(projMatrix * viewMatrix);
		UpdateShaderMatrices();

		//Return to default 'usable' state every frame!
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glDisable(GL_STENCIL_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		BuildNodeLists(root);
		SortNodeLists();
		DrawNodes();
		ClearNodeLists();
		modelMatrix = Matrix4::Scale(Vector3(1000,1000,1000));
		UpdateShaderMatrices();
		water->draw();
		DrawText("Current Frame Rate: ", Vector3(0,0,0), 16.0f, framesPerSecond/* + 0.5f*/);
		DrawText("Current Physics Rate: ", Vector3(1,20,1), 16.0f,physicsFrameRate /* + 0.5f*/);
		DrawText("Current Collisions: ", Vector3(1,40,1), 16.0f, totalCollisions/* + 0.5f*/);
		DrawText("Current Collisions: ", Vector3(1,60,1), 16.0f, totalCollisions/* + 0.5f*/);
	}
	
	
	glUseProgram(0);
	SwapBuffers();
}

void	Renderer::DrawNode(SceneNode*n)	{
	if(n->GetMesh()) {
		glUniformMatrix4fv(glGetUniformLocation(currentShader->GetProgram(), "modelMatrix"),	1,false, (float*)&(n->GetWorldTransform()*Matrix4::Scale(n->GetModelScale())));
		glUniform4fv(glGetUniformLocation(currentShader->GetProgram(), "nodeColour"),1,(float*)&n->GetColour());
		glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "useTexture"), n->GetMesh()->GetTexture());

		n->Draw(*this);
	}
}

void	Renderer::BuildNodeLists(SceneNode* from)	{
	Vector3 direction = from->GetWorldTransform().GetPositionVector() - camera->GetPosition();
	from->SetCameraDistance(Vector3::Dot(direction,direction));

	if(frameFrustum.InsideFrustum(*from)) {
		if(from->GetColour().w < 1.0f) {
			transparentNodeList.push_back(from);
		}
		else{
			nodeList.push_back(from);
		}
	}

	for(vector<SceneNode*>::const_iterator i = from->GetChildIteratorStart(); i != from->GetChildIteratorEnd(); ++i) {
		BuildNodeLists((*i));
	}
}

void	Renderer::DrawNodes()	 {
	for(vector<SceneNode*>::const_iterator i = nodeList.begin(); i != nodeList.end(); ++i ) {
		DrawNode((*i));
	}

	for(vector<SceneNode*>::const_reverse_iterator i = transparentNodeList.rbegin(); i != transparentNodeList.rend(); ++i ) {
		DrawNode((*i));
	}
}

void	Renderer::SortNodeLists()	{
	std::sort(transparentNodeList.begin(),	transparentNodeList.end(),	SceneNode::CompareByCameraDistance);
	std::sort(nodeList.begin(),				nodeList.end(),				SceneNode::CompareByCameraDistance);
}

void	Renderer::ClearNodeLists()	{
	transparentNodeList.clear();
	nodeList.clear();
}

void	Renderer::SetCamera(Camera*c) {
	camera = c;
}

void	Renderer::AddNode(SceneNode* n) {
	root->AddChild(n);
}

void	Renderer::RemoveNode(SceneNode* n) {
	root->RemoveChild(n);
}


void Renderer::DrawText(const std::string &text, const Vector3 &position, const float size, float val)
{
	SetCurrentShader(simpleShader);
	string s = text + to_string((int)val);
	TextMesh* m = new TextMesh(s, *basicText);
		
	modelMatrix = Matrix4::Translation(Vector3(position.x, height-position.y,position.z)) * Matrix4::Scale(Vector3(size,size,1));
	viewMatrix.ToIdentity();
	projMatrix = Matrix4::Orthographic(-1.0f, 1.0f, (float)width, 0.0f, (float)height, 0.0f);
	textureMatrix.ToIdentity();

	glUniform4f(glGetUniformLocation(simpleShader->GetProgram(),"nodeColour"), 1,1,1,1);
	glUniform1i(glGetUniformLocation(simpleShader->GetProgram(), "useTexture"), 1);
	glUniform1i(glGetUniformLocation(simpleShader->GetProgram(), "diffuseTex"), 0);

	UpdateShaderMatrices();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE);
	m->Draw();
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	delete m;

}