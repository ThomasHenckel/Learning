/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::cout << "Kidnapped Vehicle Project init !!!" << std::endl;
  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 20; // TODO: Set the number of particles

  for (int i = 0; i < num_particles; ++i)
  {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle p = {i,sample_x,sample_y,sample_theta,1};

    particles.push_back(p);
  }
  is_initialized = true;
  std::cout << "Kidnapped Vehicle Project init done !!!" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  //std::cout << "Kidnapped Vehicle Project prediction !!!" << std::endl;

  std::default_random_engine gen;



  for (int i = 0; i < particles.size(); ++i)
  {
    Particle p = particles[i];

    p.x = p.x + velocity/yaw_rate*(sin(p.theta+yaw_rate*delta_t)-sin(p.theta));
    p.y = p.y + velocity/yaw_rate*(cos(p.theta)-cos(p.theta+yaw_rate*delta_t));
    p.theta = p.theta + yaw_rate*delta_t;

    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    particles[i] = p;
  }
  //std::cout << "Kidnapped Vehicle Project prediction done !!!" << std::endl;

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  //std::cout << "Kidnapped Vehicle Project dataAssociation !!!" << std::endl;
  for (int i = 0; i < observations.size(); ++i)
  {
      double distance = std::numeric_limits<double>::max();
      int id = 0;
      for (int j = 0; j < predicted.size(); ++j)
      {
        double calc_dist = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);
        //std::cout << "Kidnapped Vehicle Project dataAssociation !!!" << "calc_dist: " << calc_dist << "predicted[j].x" << predicted[j].x << std::endl;
        if(calc_dist<distance){
          distance = calc_dist;
          id = j;
        } 
      }
      observations[i].id = id;
      //std::cout << "Kidnapped Vehicle Project dataAssociation !!!" << "id: " << id << "distance" << distance << std::endl;
  }
  //std::cout << "Kidnapped Vehicle Project dataAssociation done !!!" << std::endl;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  std::cout << "Kidnapped Vehicle Project updateWeights !!!" << "#sensor_range" << sensor_range << std::endl;
  for (int i = 0; i < particles.size(); ++i)
  {
    Particle p = particles[i];
    vector<LandmarkObs> observations_map;
    for (int j = 0; j < observations.size(); ++j)
    {
      LandmarkObs obs = observations[j];
      double x_part, y_part, x_obs, y_obs, theta;
      x_part = p.x;
      y_part = p.y;
      x_obs = obs.x;
      y_obs = obs.y;
      theta = p.theta;

      // transform to map x coordinate
      double x_map;
      x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);

      // transform to map y coordinate
      double y_map;
      y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
      LandmarkObs l = {obs.id,x_map,y_map};
      //std::cout << "Kidnapped Vehicle Project updateWeights !!!" << "l.x: " << l.x << " l.y: " << l.y <<std::endl;
      observations_map.push_back(l);
    }
    // find the landmarks on the map that is within range
    vector<LandmarkObs> predicted;
    
    for (int i = 0; i < map_landmarks.landmark_list.size(); ++i)
    {
      double calc_dist = dist(p.x,p.y,map_landmarks.landmark_list[i].x_f,map_landmarks.landmark_list[i].y_f);
      if(calc_dist <= sensor_range)
      {
        predicted.push_back({map_landmarks.landmark_list[i].id_i,map_landmarks.landmark_list[i].x_f,map_landmarks.landmark_list[i].y_f});
      }
    }
    std::cout << "Kidnapped Vehicle Project updateWeights #map_landmarks!!!" << predicted.size() << std::endl;
    // assosiate each observation with a landmark on the map
    
    if(predicted.size() > 0)
    {
      dataAssociation(predicted,observations_map);

      double final_weight = 0;
      // for each observation calculate the error
      for (int j = 0; j < observations_map.size(); ++j)
      {
            double sig_x, sig_y, x_obs, y_obs, mu_x, mu_y;
            sig_x = std_landmark[0];
            sig_y = std_landmark[1];
            x_obs = observations_map[j].x;
            y_obs = observations_map[j].y;
            mu_x = predicted[observations_map[j].id].x;
            mu_y = predicted[observations_map[j].id].y;
            double weight = multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
            if(weight > 0)
            {
              if(final_weight == 0)
              {
                final_weight = weight;
              }
              else
              {
                final_weight = final_weight * weight;
              }
            }

      }
      std::cout << "final_weight: " << final_weight << std::endl;
      p.weight = final_weight;
      particles[i] = p;
    }
  }
  std::cout << "Kidnapped Vehicle Project updateWeights done!!!" << std::endl;
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
  return weight;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<double> w;
  for (int i = 0; i < particles.size(); ++i)
  {
    w.push_back(particles[i].weight);
  }

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution (w.begin(), w.end());

  std::vector<Particle> particles_resampled;

  for (int i = 0; i < particles.size(); ++i) {
    int number = distribution(generator);
    particles_resampled.push_back(particles[number]);
    //std::cout << "Kidnapped Vehicle Project resample !!! number" << number << " w: " << particles[number].weight << std::endl;
  }
  particles = particles_resampled;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}