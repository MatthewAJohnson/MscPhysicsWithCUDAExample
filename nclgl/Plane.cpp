#include "Plane.h"

Plane::Plane(const Vector3 &normal, float distance, bool normalise) {
	if(normalise) {
		float length = Vector3::Dot(normal,normal);

		this->normal   = normal		/ length;
		this->distance = distance	/ length;
	
	}
	else{
		this->normal = normal;
		this->distance = distance;
	}
}

bool Plane::SphereInPlane(const Vector3 &position, float radius) const {
	
	float seperation = Vector3::Dot(position,normal)+distance;

	if(seperation < -radius) {
		return false;
	}
	/*
	if(collisionData)
	{
		collisionData->m_penetration = radius - seperation;
		collisionData->m_normal = normal;
		collisionData->m_point = s0.m_pos - normal * seperation;
	}
	*/

	return true;	
}

bool Plane::PointInPlane(const Vector3 &position) const {
	float test = Vector3::Dot(position,normal);
	float test2 = test + distance;

	if(Vector3::Dot(position,normal)+distance <= 0.0f) {
		return false;
	}

	return true;
}