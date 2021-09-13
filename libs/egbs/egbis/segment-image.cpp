/*
Copyright (C) 2015 Yasutomo Kawanishi
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include "segment-image.h"

// random color
rgb random_rgb(){ 
  rgb c;
  double r;
  
  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *l, image<float> *a, image<float> *b,image<float> *d,
			 int x1, int y1, int x2, int y2) {

  return sqrt(square(imRef(l, x1, y1)-imRef(l, x2, y2)) +
          square(imRef(a, x1, y1)-imRef(a, x2, y2)) +
          square(imRef(b, x1, y1)-imRef(b, x2, y2)) );

}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
universe *segmentation(image<lab> *im, image<float> *depth,float sigma, float c, int min_size,
			  int *num_ccs) {
  int width = im->width();
  int height = im->height();

  image<float> *l = new image<float>(width, height);
  image<float> *a = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(l, x, y) = imRef(im, x, y).l;
      imRef(a, x, y) = imRef(im, x, y).a;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_l = smooth(l, sigma);
  image<float> *smooth_a = smooth(a, sigma);
  image<float> *smooth_b = smooth(b, sigma);

  delete l;
  delete a;
  delete b;
 
  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1) {
	edges[num].a = y * width + x;
	edges[num].b = y * width + (x+1);
    edges[num].w = diff(smooth_l, smooth_a, smooth_b, depth,x, y, x+1, y);
	num++;
      }

      if (y < height-1) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + x;
    edges[num].w = diff(smooth_l, smooth_a, smooth_b, depth,x, y, x, y+1);
	num++;
      }

      if ((x < width-1) && (y < height-1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + (x+1);
    edges[num].w = diff(smooth_l, smooth_a, smooth_b, depth,x, y, x+1, y+1);
	num++;
      }

      if ((x < width-1) && (y > 0)) {
	edges[num].a = y * width + x;
	edges[num].b = (y-1) * width + (x+1);
    edges[num].w = diff(smooth_l, smooth_a, smooth_b, depth,x, y, x+1, y-1);
	num++;
      }
    }
  }
  delete smooth_l;
  delete smooth_a;
  delete smooth_b;

  // segment
  universe *u = segment_graph(width*height, num, edges, c);
  
  // post process small components
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;
  *num_ccs = u->num_sets();

  return u;
}

image<u_int16_t>* getLabelImage(universe *u, int width, int height,int num_segs){
  image<u_int16_t> *output = new image<u_int16_t>(width, height);


  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int comp = u->find(y * width + x);
      imRef(output, x, y) = (comp/(height*width*1.0f))*num_segs;
    }
  }  

  return output;
}

image<u_int16_t> *segment_image(image<lab> *im, image<float> *depth,float sigma, float c, int min_size,
			  int *num_ccs) {
    universe *u = segmentation(im, depth,sigma, c, min_size, num_ccs);
    image<u_int16_t> *lblImg = getLabelImage(u, im->width(), im->height(),*num_ccs);
	delete u;
    return lblImg;
}