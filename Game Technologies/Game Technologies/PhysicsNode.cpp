#include "PhysicsNode.h"

PhysicsNode::PhysicsNode(void)	
{
	target = NULL;
	hasGravity = true;
	isAwake = true;
	lifeSpan = rand() % 1500 + 200;
	hasLifeSpan = true;
//	type = COLLISION_SPHERE;
}

PhysicsNode::PhysicsNode(Quaternion orientation, Vector3 position) 
{
	m_orientation	= orientation;
	m_position		= position;
	m_lastPosition = m_position; // for first iteration
	m_invInertia.ToIdentity();
	hasGravity = true;
	isAwake = true;
	lifeSpan = rand() % 1500 + 200;
	hasLifeSpan = true;
//	type = COLLISION_SPHERE;
}

PhysicsNode::~PhysicsNode(void)	
{

}

//You will perform your per-object physics integration, here!
//I've added in a bit that will set the transform of the
//graphical representation of this object, too.
void	PhysicsNode::Update(float msec) 
{
	//FUN GOES HERE
	//Verlet


	if(hasGravity && m_invMass > 0.f && isAwake)
	{
		float mps = msec * 0.001f;// meters per second
		Vector3 nextPosition;
		Vector3 thisAcceleration = m_force * m_invMass  + Vector3(0,GRAVITY, 0); // f = ma, a = f/ m, a = f * inverse mass

		Vector3 displacement = (m_position - m_lastPosition)* DAMPING_FACTOR;
		if (displacement.Length() < MINIMUM_VELOCITY) 
		{
			displacement = Vector3(0,0,0);
			
			SetToRest();
			
		}

		nextPosition = m_position + displacement + thisAcceleration * mps * mps;
	
		m_lastPosition = m_position;
		m_position = nextPosition;
	
		//m_torque = distance * force 
	
		//change torque to "angular aceleration = torque / inertia" - need to calc inertia matrix
		Vector3 angularAcceleration = m_invInertia * m_torque;

		m_angularVelocity = (m_angularVelocity + angularAcceleration * mps);
		m_angularVelocity = m_angularVelocity * ANG_DAMP_FAC;
		if (m_angularVelocity.Length() < MINIMUM_ANG_VELO) 
		{
			m_angularVelocity = Vector3(0,0,0);
		
		}
		m_orientation = m_orientation + m_orientation * (m_angularVelocity * (mps/2));
		m_angularVelocity.Normalise();
		m_orientation.Normalise();
	}

	if(target)
	{
		target->SetTransform(BuildTransform());
	}
	m_force = Vector3(0,0,0);
	m_torque = Vector3(0,0,0);
}

/*
This function simply turns the orientation and position
of our physics node into a transformation matrix, suitable
for plugging into our Renderer!

It is cleaner to work with matrices when it comes to rendering,
as it is what shaders expect, and allow us to keep all of our
transforms together in a single construct. But when it comes to
physics processing and 'game-side' logic, it is much neater to
have seperate orientations and positions.

*/
Matrix4		PhysicsNode::BuildTransform() 
{
	Matrix4 m = m_orientation.ToMatrix();

	m.SetPositionVector(m_position);

	return m;
}


//stick these in a seperate helper class later
void PhysicsNode::SolidCuboidInertia(float height, float width, float length)
{
	float mass = 1 / m_invMass;
	m_invInertia[0] = 1 / ((1 / 12.0f * mass) * ((height * height) + (width * width)));
	m_invInertia[5] = 1 / ((1 / 12.0f * mass) * ((length * length) + (width * width)));
	m_invInertia[10] = 1 / ((1 / 12.0f * mass) * ((height * height) + (length * length)));
	m_invInertia[15] = 1;
}

void PhysicsNode::SolidSphereInertia(float radius)
{
	float mass = 1 / m_invMass;
	for(int i = 0; i < 16 ; ++i)
	{
		m_invInertia[i] = 0;
	}

	float I = 2.5f/(mass*radius*radius);
	//m_invInertia[0] = 1 / (((2 * mass) * (radius * radius)) / 5.0f);
	//m_invInertia[5] = 1 / (((2 * mass) * (radius * radius)) / 5.0f);
	//m_invInertia[10] = 1 / (((2 * mass) * (radius * radius)) / 5.0f);
	m_invInertia[0] = I;
	m_invInertia[5] = I;
	m_invInertia[10] = I;
	m_invInertia[15] = 1;
}

//add force method 
//vec3 contact, vec3 force, vec3 normal 
// 
// 

void PhysicsNode::EnableGravity()
{
	hasGravity = true;
}

void PhysicsNode::DisableGravity()
{
	hasGravity = false;
}

void PhysicsNode::SetToRest()
{
	isAwake = false;
}

void PhysicsNode::WakeUp()
{
	isAwake = true;
}
