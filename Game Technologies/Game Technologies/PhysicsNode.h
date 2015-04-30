/******************************************************************************
Class:PhysicsNode
Implements:
Author:Rich Davison	<richard.davison4@newcastle.ac.uk> and YOU!
Description: This class represents the physical properties of your game's
entities - their position, orientation, mass, collision volume, and so on.
Most of the first few tutorials will be based around adding code to this class
in order to add correct physical integration of velocity / acceleration etc to
your game objects. 


In addition to the tutorial code, this class contains a pointer to a SceneNode.
This pointer is to the 'graphical' representation of your game object, just 
like the SceneNode's used in the graphics module. However, instead of 
calculating positions etc as part of the SceneNode, it should instead be up
to your 'physics' representation to determine - so we calculate a transform
matrix for your SceneNode here, and apply it to the pointer. 

Your SceneNode should still have an Update function, though, in order to
update any graphical effects applied to your object - anything that will end
up modifying a uniform in a shader should still be the job of the SceneNode. 

Note that since the SceneNode can still have children, we can represent our
entire CubeRobot with a single PhysicsNode, and a single SceneNode 'root'.

-_-_-_-_-_-_-_,------,   
_-_-_-_-_-_-_-|   /\_/\   NYANYANYAN
-_-_-_-_-_-_-~|__( ^ .^) /
_-_-_-_-_-_-_-""  ""   

*//////////////////////////////////////////////////////////////////////////////


#pragma once

#include "../../nclgl/Quaternion.h"
#include "../../nclgl/Vector3.h"
#include "../../nclgl/Matrix4.h"
#include "../../nclgl/SceneNode.h"
#include "PhysicsHelper.h"
#define DAMPING_FACTOR 0.999f
#define ANG_DAMP_FAC 0.575f
#define MINIMUM_VELOCITY 0.0001f
#define MINIMUM_ANG_VELO 0.555f
#define GRAVITY -98.10f

#define RENDER_HZ	60
#define PHYSICS_HZ	120

#define PHYSICS_TIMESTEP (1000.0f / (float)PHYSICS_HZ)


class PhysicsNode	{
public:
	PhysicsNode(void);
	PhysicsNode(Quaternion orientation, Vector3 position);
	~PhysicsNode(void);

	Vector3		GetPosition()			{ return m_position;}
	void		SetPosition(Vector3 position) {m_position = position; m_lastPosition = position;}
	Vector3		GetLinearVelocity()		{ return (m_position - m_lastPosition)/(PHYSICS_TIMESTEP * 0.001f);}
	void		SetLinearVelocity(Vector3 newLinVelo){m_position = m_lastPosition + newLinVelo*PHYSICS_TIMESTEP * 0.001f;}
														
	//Vel = newPos - oldPos
	//oldPos + vel = newPos 
	


	float		GetInverseMass()		{return m_invMass;}
	void		SetInverseMass(float newInvMass){m_invMass = newInvMass;}

	Quaternion	GetOrientation()		{ return m_orientation;}
	Vector3		GetAngularVelocity()	{ return m_angularVelocity;}
	void		SetAngularVelocity(Vector3 newAngVelo){m_angularVelocity = newAngVelo;}

	Matrix4		BuildTransform();

	virtual void		Update(float msec);

	void	SetTarget(SceneNode *s) { target = s;}

	Vector3	GetForce()	{ return m_force;}
	Vector3	GetTorque() { return m_torque;}
	void	SetForce(Vector3 newForce) {m_force = newForce;}

	void SolidCuboidInertia(float height, float width, float length);
	void SolidSphereInertia(float radius);
	void ImoveableObject()
	{
		m_invMass = 0.f;
	
		m_invInertia[0] = 0.f;
		m_invInertia[5] = 0.f;
		m_invInertia[10] = 0.f;
	}

	void EnableGravity();
	void DisableGravity();

	CollisionVolume* GetCollisionVolume(){return type;}//this is wrong, :( //maybe? ;_; 
	void			 SetCollisionVolumeType(CollisionVolume* t){type = t;}

	Matrix4 GetInverseInertia(){return m_invInertia;}

	void SetToRest();
	void WakeUp();
	bool GetRestedState(){return isAwake;}

	void DecreaseLife(){ --lifeSpan;}
	void EnableLifeSpan(){hasLifeSpan = true;}
	void DisableLifeSpan(){hasLifeSpan = false;}
	int GetCurrentLifeSpan(){return lifeSpan;}
	bool HasLifeSpan(){return hasLifeSpan;}

protected:
	//<---------LINEAR-------------->
	Vector3		m_position;
	Vector3		m_linearVelocity;
	Vector3		m_force;
	float		m_invMass;

	Vector3		m_lastPosition;

	//<----------ANGULAR--------------->
	Quaternion  m_orientation;
	Vector3		m_angularVelocity;
	Vector3		m_torque;
	Matrix4     m_invInertia;

	SceneNode*	target;  

	bool isAwake;
	
	bool hasGravity;

	bool hasLifeSpan;
	int lifeSpan;

	CollisionVolume* type;


};

