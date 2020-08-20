#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits.h>
#include <float.h>
#include <iostream>
#include "lapack.h"  //matlab 
//#include "include/lapacke_config.h"  //lapack手动，未成功
//#include "include/lapacke.h"
#include "opencv2/core/core.hpp" 
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2\opencv.hpp>
using namespace cv;



#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0
/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */
#define M_1_2_PI 1.57079632679489661923
#define M_1_4_PI 0.785398163

#define M_3_4_PI 2.35619449

#define M_1_8_PI 0.392699081
#define M_3_8_PI 1.178097245
#define M_5_8_PI 1.963495408
#define M_7_8_PI 2.748893572
#define M_4_9_PI 1.396263401595464  //80°
#define M_1_9_PI  0.34906585  //20°
#define M_1_10_PI 0.314159265358979323846   //18°
#define M_1_12_PI 0.261799387   //15°
#define M_1_15_PI 0.20943951    //12°
#define M_1_18_PI 0.174532925   //10°
/** 3/2 pi */
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI  6.28318530718
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

struct point2i //(or pixel).
{
	int x,y;
};

struct point2d
{
	double x,y;
};

struct point1d1i
{
	double data;
	int cnt;
};

struct point3d
{
	double x,y;
	double r;
};

struct point3i
{
	int x,y;
	int z;
};

struct point2d1i
{
	double x,y;
	int z;
};

struct  point5d
{
	double x,y;
	double a,b;
	double phi;
};

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
  double x1,y1,x2,y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x,y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx,dy;        /* (dx,dy) is vector oriented as the line segment,dx = cos(theta), dy = sin(theta) */
  int   polarity;     /* if the arc direction is the same as the edge direction, polarity = 1, else if opposite ,polarity = -1.*/
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
};

typedef struct
{
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  double ys,ye;  /* start and end Y values of current 'column' */
  int x,y;       /* coordinates of currently explored pixel */
} rect_iter;

typedef struct image_double_s
{
  double * data;
  int xsize,ysize;
} * image_double;


//==================================================================================================
//=============================miscellaneous functions==============================================
inline double min(double v1,double v2)
{
	return (v1<v2?v1:v2);
}
inline double max(double v1,double v2)
{
	return (v1>v2?v1:v2);
}
/** Compare doubles by relative error.

    The resulting rounding error after floating point computations
    depend on the specific operations done. The same number computed by
    different algorithms could present different rounding errors. For a
    useful comparison, an estimation of the relative rounding error
    should be considered and compared to a factor times EPS. The factor
    should be related to the cumulated rounding error in the chain of
    computation. Here, as a simplification, a fixed factor is used.
 */
int double_equal(double a, double b)
{
  double abs_diff,aa,bb,abs_max;

  /* trivial case */
  if( a == b ) return TRUE;

  abs_diff = fabs(a-b);
  aa = fabs(a);
  bb = fabs(b);
  abs_max = aa > bb ? aa : bb;

  /* DBL_MIN is the smallest normalized number, thus, the smallest
     number whose relative error is bounded by DBL_EPSILON. For
     smaller numbers, the same quantization steps as for DBL_MIN
     are used. Then, for smaller numbers, a meaningful "relative"
     error should be computed by dividing the difference by DBL_MIN. */
  if( abs_max < DBL_MIN ) abs_max = DBL_MIN;

  /* equal if relative error <= factor x eps */
  return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON); //RELATIVE_ERROR_FACTOR=100.0,
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
//得到2个弧度制角度的夹角的绝对值
double angle_diff(double a, double b)
{
  a -= b;
  while( a <= -M_PI ) a += M_2__PI;
  while( a >   M_PI ) a -= M_2__PI;
  if( a < 0.0 ) a = -a;
  return a;
}
/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
double angle_diff_signed(double a, double b)
{
  a -= b;
  while( a <= -M_PI ) a += M_2__PI;
  while( a >   M_PI ) a -= M_2__PI;
  return a;
}

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
void error(char * msg)
{
  fprintf(stderr,"circleDetection Error: %s\n",msg);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */
double dist(double x1, double y1, double x2, double y2)
{
  return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
}

//向量内积
double dotProduct(point2d vec1, point2d vec2)
{
	return (vec1.x*vec2.x+vec1.y*vec2.y);
}

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
void rect_copy(struct rect * in, struct rect * out)//in is the src, out is the dst
{
  /* check parameters */
  if( in == NULL || out == NULL ) error("rect_copy: invalid 'in' or 'out'.");

  /* copy values */
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->width = in->width;
  out->x = in->x;
  out->y = in->y;
  out->theta = in->theta;
  out->dx = in->dx;
  out->dy = in->dy;
  out->polarity = in->polarity;
  out->prec = in->prec;
  out->p = in->p;
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the smaller
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
double inter_low(double x, double x1, double y1, double x2, double y2)
{
  /* check parameters */
  if( x1 > x2 || x < x1 || x > x2 )
    error("inter_low: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if( double_equal(x1,x2) && y1<y2 ) return y1;
  if( double_equal(x1,x2) && y1>y2 ) return y2;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the larger
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
double inter_hi(double x, double x1, double y1, double x2, double y2)
{
  /* check parameters */
  if( x1 > x2 || x < x1 || x > x2 )
    error("inter_hi: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if( double_equal(x1,x2) && y1<y2 ) return y2;
  if( double_equal(x1,x2) && y1>y2 ) return y1;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
void ri_del(rect_iter * iter)
{
  if( iter == NULL ) error("ri_del: NULL iterator.");
  free( (void *) iter );
}

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

    See details in \ref rect_iter
 */
int ri_end(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_end: NULL iterator.");

  /* if the current x value is larger than the largest
     x value in the rectangle (vx[2]), we know the full
     exploration of the rectangle is finished. */
  return (double)(i->x) > i->vx[2];
}

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

    See details in \ref rect_iter
 */
void ri_inc(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");

  /* if not at end of exploration,
     increase y value for next pixel in the 'column' */
  if( !ri_end(i) ) i->y++;

  /* if the end of the current 'column' is reached,
     and it is not the end of exploration,
     advance to the next 'column' */
  while( (double) (i->y) > i->ye && !ri_end(i) )
    {
      /* increase x, next 'column' */
      i->x++;

      /* if end of exploration, return */
      if( ri_end(i) ) return;

      /* update lower y limit (start) for the new 'column'.

         We need to interpolate the y value that corresponds to the
         lower side of the rectangle. The first thing is to decide if
         the corresponding side is

           vx[0],vy[0] to vx[3],vy[3] or
           vx[3],vy[3] to vx[2],vy[2]

         Then, the side is interpolated for the x value of the
         'column'. But, if the side is vertical (as it could happen if
         the rectangle is vertical and we are dealing with the first
         or last 'columns') then we pick the lower value of the side
         by using 'inter_low'.
       */
      if( (double) i->x < i->vx[3] )
        i->ys = inter_low((double)i->x,i->vx[0],i->vy[0],i->vx[3],i->vy[3]);
      else
        i->ys = inter_low((double)i->x,i->vx[3],i->vy[3],i->vx[2],i->vy[2]);

      /* update upper y limit (end) for the new 'column'.

         We need to interpolate the y value that corresponds to the
         upper side of the rectangle. The first thing is to decide if
         the corresponding side is

           vx[0],vy[0] to vx[1],vy[1] or
           vx[1],vy[1] to vx[2],vy[2]

         Then, the side is interpolated for the x value of the
         'column'. But, if the side is vertical (as it could happen if
         the rectangle is vertical and we are dealing with the first
         or last 'columns') then we pick the lower value of the side
         by using 'inter_low'.
       */
      if( (double)i->x < i->vx[1] )
        i->ye = inter_hi((double)i->x,i->vx[0],i->vy[0],i->vx[1],i->vy[1]);
      else
        i->ye = inter_hi((double)i->x,i->vx[1],i->vy[1],i->vx[2],i->vy[2]);

      /* new y */
      i->y = (int) ceil(i->ys);
    }
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
rect_iter * ri_ini(struct rect * r)
{
  double vx[4],vy[4];
  int n,offset;
  rect_iter * i;

  /* check parameters */
  if( r == NULL ) error("ri_ini: invalid rectangle.");

  /* get memory */
  i = (rect_iter *) malloc(sizeof(rect_iter));
  if( i == NULL ) error("ri_ini: Not enough memory.");

  /* build list of rectangle corners ordered
     in a circular way around the rectangle */
  //从线段的起点(x1,y1)处的一端开始按照逆时针重构出矩形的四个定点
  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  /* compute rotation of index of corners needed so that the first
     point has the smaller x.

     if one side is vertical, thus two corners have the same smaller x
     value, the one with the largest y value is selected as the first.
   */
  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;

  /* apply rotation of index. */
  for(n=0; n<4; n++)
    {
      i->vx[n] = vx[(offset+n)%4];
      i->vy[n] = vy[(offset+n)%4];
    }

  /* Set an initial condition.

     The values are set to values that will cause 'ri_inc' (that will
     be called immediately) to initialize correctly the first 'column'
     and compute the limits 'ys' and 'ye'.

     'y' is set to the integer value of vy[0], the starting corner.

     'ys' and 'ye' are set to very small values, so 'ri_inc' will
     notice that it needs to start a new 'column'.

     The smallest integer coordinate inside of the rectangle is
     'ceil(vx[0])'. The current 'x' value is set to that value minus
     one, so 'ri_inc' (that will increase x by one) will advance to
     the first 'column'.
   */
  i->x = (int) ceil(i->vx[0]) - 1;
  i->y = (int) ceil(i->vy[0]);
  i->ys = i->ye = -DBL_MAX;

  /* advance to the first pixel */
  ri_inc(i);

  return i;
}


/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
void free_image_double(image_double i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_double: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
image_double new_image_double(int xsize, int ysize)
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_double: invalid image size.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (double *) calloc( (size_t) (xsize*ysize), sizeof(double) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
image_double new_image_double_ptr( int xsize,
                                          int ysize, double * data )
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 )
    error("new_image_double_ptr: invalid image size.");
  if( data == NULL ) error("new_image_double_ptr: NULL data pointer.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  image->data = data;

  return image;
}

//=================================================================================================================
//===========================================LSD functions=========================================================
/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402    //ln10
#endif /* !M_LN10 */

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED    1

//对于构成圆弧的像素标记极性，如果梯度的方向和弧的方向指向一致，则为SAME_POLE,否则为OPP_POLE,该标记初始是为0
#define NOTDEF_POL 0
#define SAME_POL 1
#define OPP_POL  -1
/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist
{
  int x,y;
  struct coorlist * next;
};
typedef struct ntuple_list_s
{
  int size;
  int max_size;
  int dim;
  double * values;
} * ntuple_list;

/*----------------------------------------------------------------------------*/
/** Free memory used in n-tuple 'in'.
 */
static void free_ntuple_list(ntuple_list in)
{
  if( in == NULL || in->values == NULL )
    error("free_ntuple_list: invalid n-tuple input.");
  free( (void *) in->values );
  free( (void *) in );
}

/*----------------------------------------------------------------------------*/
/** Create an n-tuple list and allocate memory for one element.
    @param dim the dimension (n) of the n-tuple.
 */
static ntuple_list new_ntuple_list(int dim)
{
  ntuple_list n_tuple;

  /* check parameters */
  if( dim == 0 ) error("new_ntuple_list: 'dim' must be positive.");

  /* get memory for list structure */
  n_tuple = (ntuple_list) malloc( sizeof(struct ntuple_list_s) );
  if( n_tuple == NULL ) error("not enough memory.");

  /* initialize list */
  n_tuple->size = 0;
  n_tuple->max_size = 1;
  n_tuple->dim = dim;

  /* get memory for tuples */
  n_tuple->values = (double *) malloc( dim*n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");

  return n_tuple;
}

/*----------------------------------------------------------------------------*/
/** Enlarge the allocated memory of an n-tuple list.
 */
static void enlarge_ntuple_list(ntuple_list n_tuple)
{
  /* check parameters */
  if( n_tuple == NULL || n_tuple->values == NULL || n_tuple->max_size == 0 )
    error("enlarge_ntuple_list: invalid n-tuple.");

  /* duplicate number of tuples */
  n_tuple->max_size *= 2;

  /* realloc memory */
  n_tuple->values = (double *) realloc( (void *) n_tuple->values,
                      n_tuple->dim * n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");
}

/*----------------------------------------------------------------------------*/
/** Add a 7-tuple to an n-tuple list.
 */
static void add_7tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7 )
{
  /* check parameters */
  if( out == NULL ) error("add_7tuple: invalid n-tuple input.");
  if( out->dim != 7 ) error("add_7tuple: the n-tuple must be a 7-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_7tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;

  /* update number of tuples counter */
  out->size++;
}
/*----------------------------------------------------------------------------*/
/** Add a 8-tuple to an n-tuple list.
 */
static void add_8tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7, int v8)
{
  /* check parameters */
  if( out == NULL ) error("add_8tuple: invalid n-tuple input.");
  if( out->dim != 8 ) error("add_8tuple: the n-tuple must be a 8-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_8tuple: invalid n-tuple input.");

  /* add new 8-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;
  out->values[ out->size * out->dim + 7 ] = v8;

  /* update number of tuples counter */
  out->size++;
}
/** char image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize;
} * image_char;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
static void free_image_char(image_char i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_char: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
static image_char new_image_char(unsigned int xsize, unsigned int ysize)
{
  image_char image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_char: invalid image size.");

  /* get memory */
  image = (image_char) malloc( sizeof(struct image_char_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (unsigned char *) calloc( (size_t) (xsize*ysize),
                                          sizeof(unsigned char) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_char new_image_char_ini( unsigned int xsize, unsigned int ysize,
                                      unsigned char fill_value )
{
  image_char image = new_image_char(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* check parameters */
  if( image == NULL || image->data == NULL )
    error("new_image_char_ini: invalid image.");

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** int image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_int_s
{
  int * data;
  unsigned int xsize,ysize;
} * image_int;

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
static image_int new_image_int(unsigned int xsize, unsigned int ysize)
{
  image_int image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_int: invalid image size.");

  /* get memory */
  image = (image_int) malloc( sizeof(struct image_int_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (int *) calloc( (size_t) (xsize*ysize), sizeof(int) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_int new_image_int_ini( unsigned int xsize, unsigned int ysize,
                                    int fill_value )
{
  image_int image = new_image_int(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}
/** Compute a Gaussian kernel of length 'kernel->dim',
    standard deviation 'sigma', and centered at value 'mean'.

    For example, if mean=0.5, the Gaussian will be centered
    in the middle point2i between values 'kernel->values[0]'
    and 'kernel->values[1]'.
 */
static void gaussian_kernel(ntuple_list kernel, double sigma, double mean)
{
  double sum = 0.0;
  double val;
  int i;

  /* check parameters */
  if( kernel == NULL || kernel->values == NULL )
    error("gaussian_kernel: invalid n-tuple 'kernel'.");
  if( sigma <= 0.0 ) error("gaussian_kernel: 'sigma' must be positive.");

  /* compute Gaussian kernel */
  if( kernel->max_size < 1 ) enlarge_ntuple_list(kernel);
  kernel->size = 1;
  for(i=0;i<kernel->dim;i++)
    {
      val = ( (double) i - mean ) / sigma;
      kernel->values[i] = exp( -0.5 * val * val );
      sum += kernel->values[i];
    }

  /* normalization */
  if( sum >= 0.0 ) for(i=0;i<kernel->dim;i++) kernel->values[i] /= sum;
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

    For example, scale=0.8 will give a result at 80% of the original size.

    The image is convolved with a Gaussian kernel
    @f[
        G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
    @f]
    before the sub-sampling to prevent aliasing.

    The standard deviation sigma given by:
    -  sigma = sigma_scale / scale,   if scale <  1.0
    -  sigma = sigma_scale,           if scale >= 1.0

    To be able to sub-sample at non-integer steps, some interpolation
    is needed. In this implementation, the interpolation is done by
    the Gaussian kernel, so both operations (filtering and sampling)
    are done at the same time. The Gaussian kernel is computed
    centered on the coordinates of the required sample. In this way,
    when applied, it gives directly the result of convolving the image
    with the kernel and interpolated to that particular position.

    A fast algorithm is done using the separability of the Gaussian
    kernel. Applying the 2D Gaussian kernel is equivalent to applying
    first a horizontal 1D Gaussian kernel and then a vertical 1D
    Gaussian kernel (or the other way round). The reason is that
    @f[
        G(x,y) = G(x) * G(y)
    @f]
    where
    @f[
        G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
    @f]
    The algorithm first applies a combined Gaussian kernel and sampling
    in the x axis, and then the combined Gaussian kernel and sampling
    in the y axis.
 */
static image_double gaussian_sampler( image_double in, double scale,
                                      double sigma_scale )
{
  image_double aux,out;
  ntuple_list kernel;
  int N,M,h,n,x,y,i;
  int xc,yc,j,double_x_size,double_y_size;
  double sigma,xx,yy,sum,prec;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("gaussian_sampler: invalid image.");
  if( scale <= 0.0 ) error("gaussian_sampler: 'scale' must be positive.");
  if( sigma_scale <= 0.0 )
    error("gaussian_sampler: 'sigma_scale' must be positive.");

  /* compute new image size and get memory for images */
  if( in->xsize * scale > (double) UINT_MAX ||
      in->ysize * scale > (double) UINT_MAX )
    error("gaussian_sampler: the output image size exceeds the handled size.");
  N = (unsigned int) ceil( in->xsize * scale );//上取整
  M = (unsigned int) ceil( in->ysize * scale );
  aux = new_image_double(N,in->ysize);
  out = new_image_double(N,M);

  /* sigma, kernel size and memory for the kernel */
  sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale;
  /*
     The size of the kernel is selected to guarantee that the
     the first discarded term is at least 10^prec times smaller
     than the central value. For that, h should be larger than x, with
       e^(-x^2/2sigma^2) = 1/10^prec.
     Then,
       x = sigma * sqrt( 2 * prec * ln(10) ).
   */
  prec = 3.0;//高斯核的最外围降到10^(-3)
  h = (unsigned int) ceil( sigma * sqrt( 2.0 * prec * log(10.0) ) );
  n = 1+2*h; /* kernel size */
  kernel = new_ntuple_list(n);

  /* auxiliary double image size variables */
  double_x_size = (int) (2 * in->xsize);
  double_y_size = (int) (2 * in->ysize);

  /* First subsampling: x axis */
  for(x=0;x<aux->xsize;x++)
    {
      /*
         x   is the coordinate in the new image.
         xx  is the corresponding x-value in the original size image.
         xc  is the integer value, the pixel coordinate of xx.
       */
      xx = (double) x / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with xc=0 get the values of xx from -0.5 to 0.5 */
      xc = (int) floor( xx + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + xx - (double) xc );
      /* the kernel must be computed for each x because the fine
         offset xx-xc is different in each case */

      for(y=0;y<aux->ysize;y++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = xc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_x_size;
              while( j >= double_x_size ) j -= double_x_size;
              if( j >= (int) in->xsize ) j = double_x_size-1-j;

              sum += in->data[ j + y * in->xsize ] * kernel->values[i];
            }
          aux->data[ x + y * aux->xsize ] = sum;
        }
    }

  /* Second subsampling: y axis */
  for(y=0;y<out->ysize;y++)
    {
      /*
         y   is the coordinate in the new image.
         yy  is the corresponding x-value in the original size image.
         yc  is the integer value, the pixel coordinate of xx.
       */
      yy = (double) y / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with yc=0 get the values of yy from -0.5 to 0.5 */
      yc = (int) floor( yy + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + yy - (double) yc );
      /* the kernel must be computed for each y because the fine
         offset yy-yc is different in each case */

      for(x=0;x<out->xsize;x++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = yc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_y_size;
              while( j >= double_y_size ) j -= double_y_size;
              if( j >= (int) in->ysize ) j = double_y_size-1-j;

              sum += aux->data[ x + j * aux->xsize ] * kernel->values[i];
            }
          out->data[ x + y * out->xsize ] = sum;
        }
    }

  /* free memory */
  free_ntuple_list(kernel);
  free_image_double(aux);

  return out;
}


/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point2i.

    The result is:
    - an image_double with the angle at each pixel, or NOTDEF if not defined.
    - the image_double 'modgrad' (a point2ier is passed as argument)
      with the gradient magnitude at each point2i.
    - a list of pixels 'list_p' roughly ordered by decreasing
      gradient magnitude. (The order is made by classifying point2is
      into bins by gradient magnitude. The parameters 'n_bins' and
      'max_grad' specify the number of bins and the gradient modulus
      at the highest bin. The pixels in the list would be in
      decreasing gradient magnitude, up to a precision of the size of
      the bins.)
    - a point2ier 'mem_p' to the memory used by 'list_p' to be able to
      free the memory when it is not used anymore.
 */
//返回一张梯度角度顺时针旋转90°后的align角度图angles，如果梯度角度是(gx,gy)->(-gy,gx)，
//和梯度的模的图modgrad,然后按照n_bins进行伪排序返回链表的头指针list_p,里面存的是坐标
static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p,
                              image_double * modgrad, unsigned int n_bins )
{
  image_double g;
  unsigned int n,p,x,y,adr,i;
  double com1,com2,gx,gy,norm,norm2;
  /* the rest of the variables are used for pseudo-ordering
     the gradient magnitude values */
  int list_count = 0;
  //struct coorlist * list;
  struct coorlist *temp;
  struct coorlist ** range_l_s; /* array of point2iers to start of bin list,表示1024个bin的头指针的指针数组 */
  struct coorlist ** range_l_e; /* array of point2iers to end of bin list，表示1024个bin的尾指针的指针数组*/
  struct coorlist * start;
  struct coorlist * end;
  double max_grad = 0.0;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("ll_angle: invalid image.");
  if( threshold < 0.0 ) error("ll_angle: 'threshold' must be positive.");
  if( list_p == NULL ) error("ll_angle: NULL point2ier 'list_p'.");
 // if( mem_p == NULL ) error("ll_angle: NULL point2ier 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle: NULL point2ier 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle: 'n_bins' must be positive.");

  /* image size shortcuts */
  n = in->ysize;
  p = in->xsize;

  /* allocate output image */
  g = new_image_double(in->xsize,in->ysize);

  /* get memory for the image of gradient modulus */
  *modgrad = new_image_double(in->xsize,in->ysize);

  /* get memory for "ordered" list of pixels */
  //list = (struct coorlist *) calloc( (size_t) (n*p), sizeof(struct coorlist) );
  //*mem_p = (void *) list;
  range_l_s = (struct coorlist **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
 // if( list == NULL || range_l_s == NULL || range_l_e == NULL )
  if( range_l_s == NULL || range_l_e == NULL )
    error("not enough memory.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;

  /* 'undefined' on the down and right boundaries */
  for(x=0;x<p;x++) g->data[(n-1)*p+x] = NOTDEF;// p = in->xsize
  for(y=0;y<n;y++) g->data[p*y+p-1]   = NOTDEF;// n = in->ysize;

  /* compute gradient on the remaining pixels */
  for(x=0;x<p-1;x++)
    for(y=0;y<n-1;y++)
      {
        adr = y*p+x;

        /*
           Norm 2 computation using 2x2 pixel window:
             A B
             C D
           and
             com1 = D-A,  com2 = B-C.
           Then
             gx = B+D - (A+C)   horizontal difference
             gy = C+D - (A+B)   vertical difference
           com1 and com2 are just to avoid 2 additions.
         */
        com1 = in->data[adr+p+1] - in->data[adr];
        com2 = in->data[adr+1]   - in->data[adr+p];

        gx = com1+com2; /* gradient x component */
        gy = com1-com2; /* gradient y component */
        norm2 = gx*gx+gy*gy;
        norm = sqrt( norm2 / 4.0 ); /* gradient norm */

        (*modgrad)->data[adr] = norm; /* store gradient norm */

        if( norm <= threshold ) /* norm too small, gradient no defined */
          g->data[adr] = NOTDEF; /* gradient angle not defined */
        else
          {
            /* gradient angle computation */
            g->data[adr] = atan2(gx,-gy);

            /* look for the maximum of the gradient */
            if( norm > max_grad ) max_grad = norm;
          }
      }

  /* compute histogram of gradient values */
  for(x=0;x<p-1;x++)
    for(y=0;y<n-1;y++)
      {
		temp = new coorlist();
		if(temp == NULL)
		{
			printf("not enough memory");
			system("pause");
		}
        norm = (*modgrad)->data[y*p+x];
        /* store the point2i in the right bin according to its norm */
        i = (unsigned int) (norm * (double) n_bins / max_grad);
        if( i >= n_bins ) i = n_bins-1;
        if( range_l_e[i] == NULL )
          range_l_s[i] = range_l_e[i] = temp;//记录第i个区域的头指针到range_l_s[i]
        else
          {
            range_l_e[i]->next = temp;//第i个区域由尾指针range_l_e[i]完成勾链
            range_l_e[i] = temp;
          }
        range_l_e[i]->x = (int) x;//将坐标(x,y)记录到第i个分区
        range_l_e[i]->y = (int) y;
        range_l_e[i]->next = NULL;
      }

  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  for(i=n_bins-1; i>0 && range_l_s[i]==NULL; i--);//找到第一个不为空的分区bin
  start = range_l_s[i];
  end = range_l_e[i];
  if( start != NULL )
    while(i>0)
      {
        --i;
        if( range_l_s[i] != NULL )
          {
            end->next = range_l_s[i];
            end = range_l_e[i];
          }
      }
  *list_p = start;
 // *mem_p  = start;
  /* free memory */
  free( (void *) range_l_s );
  free( (void *) range_l_e );

  return g;
}
/*----------------------------------------------------------------------------*/
/** Is point2i (x,y) aligned to angle theta, up to precision 'prec'?
 */
static int isaligned( int x, int y, image_double angles, double theta,
                      double prec )
{
  double a;

  /* check parameters */
  if( angles == NULL || angles->data == NULL )
    error("isaligned: invalid image 'angles'.");
  if( x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize )
    error("isaligned: (x,y) out of the image.");
  if( prec < 0.0 ) error("isaligned: 'prec' must be positive.");

  /* angle at pixel (x,y) */
  a = angles->data[ x + y * angles->xsize ];

  /* pixels whose level-line angle is not defined
     are considered as NON-aligned */
  if( a == NOTDEF ) return FALSE;  /* there is no need to call the function
                                      'double_equal' here because there is
                                      no risk of problems related to the
                                      comparison doubles, we are only
                                      interested in the exact NOTDEF value */

  /* it is assumed that 'theta' and 'a' are in the range [-pi,pi] */
  theta -= a;
  if( theta < 0.0 ) theta = -theta;
  if( theta > M_3_2_PI )
    {
	//--------------------------------------
	//origin code
     /* theta -= M_2__PI;
      if( theta < 0.0 ) theta = -theta;*/
	//--------------------------------------
	  //-------------------------------------
	  //mycode
	  theta = M_2__PI-theta;
	  if(theta < 0.0) 
		 theta = -theta; 
	  //--------------------------------------
    }

  return theta <= prec;
}


/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using the Lanczos approximation.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
      \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                  (x+5.5)^{x+0.5} e^{-(x+5.5)}
    @f]
    so
    @f[
      \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                      + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
    @f]
    and
      q0 = 75122.6331530,
      q1 = 80916.6278952,
      q2 = 36308.2951477,
      q3 = 8687.24529705,
      q4 = 1168.92649479,
      q5 = 83.8676043424,
      q6 = 2.50662827511.
 */
static double log_gamma_lanczos(double x)
{
  static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
  double a = (x+0.5) * log(x+5.5) - (x+5.5);
  double b = 0.0;
  int n;

  for(n=0;n<7;n++)
    {
      a -= log( x + (double) n );
      b += q[n] * pow( x, (double) n );
    }
  return a + log(b);
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using Windschitl method.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
        \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                    \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
    @f]
    so
    @f[
        \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                      + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
    @f]
    This formula is a good approximation when x > 15.
 */
static double log_gamma_windschitl(double x)
{
  return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x. When x>15 use log_gamma_windschitl(),
    otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/** Size of the table to store already computed inverse values.
 */
#define TABSIZE 100000

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}
    @f]

    The value -log10(NFA) is equivalent but more intuitive than NFA:
    - -1 corresponds to 10 mean false alarms
    -  0 corresponds to 1 mean false alarm
    -  1 corresponds to 0.1 mean false alarms
    -  2 corresponds to 0.01 mean false alarms
    -  ...

    Used this way, the bigger the value, better the detection,
    and a logarithmic scale is used.

    @param n,k,p binomial parameters.
    @param logNT logarithm of Number of Tests

    The computation is based in the gamma function by the following
    relation:
    @f[
        \left(\begin{array}{c}n\\k\end{array}\right)
        = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
    @f]
    We use efficient algorithms to compute the logarithm of
    the gamma function.

    To make the computation faster, not all the sum is computed, part
    of the terms are neglected based on a bound to the error obtained
    (an error of 10% in the result is accepted).
 */
static double nfa(int n, int k, double p, double logNT)
{
  static double inv[TABSIZE];   /* table to keep computed inverse values */
  double tolerance = 0.1;       /* an error of 10% in the result is accepted */
  double log1term,term,bin_term,mult_term,bin_tail,err,p_term;
  int i;

  /* check parameters */
  if( n<0 || k<0 || k>n || p<=0.0 || p>=1.0 )
    error("nfa: wrong n, k or p values.");

  /* trivial cases */
  if( n==0 || k==0 ) return -logNT;
  if( n==k ) return -logNT - (double) n * log10(p);

  /* probability term */
  p_term = p / (1.0-p);

  /* compute the first term of the series */
  /*
     binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
     where bincoef(n,i) are the binomial coefficients.
     But
       bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
     We use this to compute the first term. Actually the log of it.
   */
  log1term = log_gamma( (double) n + 1.0 ) - log_gamma( (double) k + 1.0 )
           - log_gamma( (double) (n-k) + 1.0 )
           + (double) k * log(p) + (double) (n-k) * log(1.0-p);
  term = exp(log1term);

  /* in some cases no more computations are needed */
  if( double_equal(term,0.0) )              /* the first term is almost zero */
    {
      if( (double) k > (double) n * p )     /* at begin or end of the tail?  */
        return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
      else
        return -logNT;                      /* begin: the tail is roughly 1  */
    }

  /* compute more terms if needed */
  bin_tail = term;
  for(i=k+1;i<=n;i++)
    {
      /*
         As
           term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
         and
           bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
         then,
           term_i / term_i-1 = (n-i+1)/i * p/(1-p)
         and
           term_i = term_i-1 * (n-i+1)/i * p/(1-p).
         1/i is stored in a table as they are computed,
         because divisions are expensive.
         p/(1-p) is computed only once and stored in 'p_term'.
       */
      bin_term = (double) (n-i+1) * ( i<TABSIZE ?
                   ( inv[i]!=0.0 ? inv[i] : ( inv[i] = 1.0 / (double) i ) ) :
                   1.0 / (double) i );

      mult_term = bin_term * p_term;
      term *= mult_term;
      bin_tail += term;
      if(bin_term<1.0)
        {
          /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
             Then, the error on the binomial tail when truncated at
             the i term can be bounded by a geometric series of form
             term_i * sum mult_term_i^j.                            */
          err = term * ( ( 1.0 - pow( mult_term, (double) (n-i+1) ) ) /
                         (1.0-mult_term) - 1.0 );

          /* One wants an error at most of tolerance*final_result, or:
             tolerance * abs(-log10(bin_tail)-logNT).
             Now, the error that can be accepted on bin_tail is
             given by tolerance*final_result divided by the derivative
             of -log10(x) when x=bin_tail. that is:
             tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
             Finally, we truncate the tail if the error is less than:
             tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
          if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
        }
    }
  double nfavalue = -log10(bin_tail) - logNT;
  return nfavalue;
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
static double rect_nfa(struct rect * rec, image_double angles, double logNT)
{
  rect_iter * i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if( rec == NULL ) error("rect_nfa: invalid rectangle.");
  if( angles == NULL ) error("rect_nfa: invalid 'angles'.");

  /* compute the total number of pixels and of aligned point2is in 'rec' */
  for(i=ri_ini(rec); !ri_end(i); ri_inc(i)) /* rectangle iterator */
    if( i->x >= 0 && i->y >= 0 &&
        i->x < (int) angles->xsize && i->y < (int) angles->ysize )
      {
        ++pts; /* total number of pixels counter */
        if( isaligned(i->x, i->y, angles, rec->theta, rec->prec) )
          ++alg; /* aligned point2is counter */
      }
  ri_del(i); /* delete iterator */
  double NFAvalue = nfa(pts,alg,rec->p,logNT); /* compute NFA value */
  return NFAvalue;
}
/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.

    The following is the region inertia matrix A:
    @f[

        A = \left(\begin{array}{cc}
                                    Ixx & Ixy \\
                                    Ixy & Iyy \\
             \end{array}\right)

    @f]
    where

      Ixx =   sum_i G(i).(y_i - cx)^2

      Iyy =   sum_i G(i).(x_i - cy)^2

      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)

    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i and y_i are the coordinates of pixel i.
    - cx and cy are the coordinates of the center of th region.

    lambda1 and lambda2 are the eigenvalues of matrix A,
    with lambda1 >= lambda2. They are found by solving the
    characteristic polynomial:

      det( lambda I - A) = 0

    that gives:

      lambda1 = ( Ixx + Iyy + sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

      lambda2 = ( Ixx + Iyy - sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

    To get the line segment direction we want to get the angle the
    eigenvector associated to the smallest eigenvalue. We have
    to solve for a,b in:

      a.Ixx + b.Ixy = a.lambda2

      a.Ixy + b.Iyy = b.lambda2

    We want the angle theta = atan(b/a). It can be computed with
    any of the two equations:

      theta = atan( (lambda2-Ixx) / Ixy )

    or

      theta = atan( Ixy / (lambda2-Iyy) )

    When |Ixx| > |Iyy| we use the first, otherwise the second (just to
    get better numeric precision).
 */
static double get_theta( point2i * reg, int reg_size, double x, double y,
                         image_double modgrad, double reg_angle, double prec )
{
  double lambda,theta,weight;
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Ixy = 0.0;
  double temp1,temp2;
  int i;

  /* check parameters */
  if( reg == NULL ) error("get_theta: invalid region.");
  if( reg_size <= 1 ) error("get_theta: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("get_theta: invalid 'modgrad'.");
  if( prec < 0.0 ) error("get_theta: 'prec' must be positive.");

  /* compute inertia matrix */
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].x + reg[i].y * modgrad->xsize ];
      Ixx += ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) * weight;
      Iyy += ( (double) reg[i].x - x ) * ( (double) reg[i].x - x ) * weight;
      Ixy -= ( (double) reg[i].x - x ) * ( (double) reg[i].y - y ) * weight;
    }
  if( double_equal(Ixx,0.0) && double_equal(Iyy,0.0) && double_equal(Ixy,0.0) )//判断Ixx、Iyy、Ixy与0是否非常接近，由于它们为double类型，故需要专门的函数判断
    error("get_theta: null inertia matrix.");

  /* compute smallest eigenvalue */
  lambda = 0.5 * ( Ixx + Iyy - sqrt( (Ixx-Iyy)*(Ixx-Iyy) + 4.0*Ixy*Ixy ) );

  /* compute angle */
  theta = fabs(Ixx)>fabs(Iyy) ? atan2(lambda-Ixx,Ixy) : atan2(Ixy,lambda-Iyy);
  /* The previous procedure doesn't cares about orientation,
     so it could be wrong by 180 degrees. Here is corrected if necessary. */
  temp1 = angle_diff(theta,reg_angle);
  if( temp1 > prec )//这是由于用惯性矩阵算出的两个正交轴的较小特征值对应的角度和该区域的角度可能相差180°
  {
	  //------------------------------------------
	  //theta += M_PI;   //origin code
	  //------------------------------------------
	  //------------------------------------------
	  //my code,增加该段代码，限制theta在 (-pi,pi)之间
	  //int flag = 0;
	  temp2 = angle_diff(theta+M_PI,reg_angle);
	  if(temp2 < prec)
	  {
		  theta += M_PI;
		if(theta > M_PI)
		{
		   theta -= M_2__PI;
		   //flag = 1;
		   //if(angle_diff(theta,reg_angle) > prec)
		   //{
		   //	  //flag = 2;
		   //	  theta = reg_angle;
		   // }
		}
	  }
	  else
	  {
		  theta = (temp2 <= temp1) ? (theta+M_PI) : theta;
		  while( theta <= -M_PI ) theta += M_2__PI;
          while( theta >   M_PI ) theta -= M_2__PI;
	  }
	  
	  //--------------------------------------------
  }
  return theta;
}

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of point2is.
 */
static void region2rect( point2i * reg, int reg_size,
						image_double modgrad, double reg_angle,
                         double prec, double p, struct rect * rec )
{
  double x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
  int i;

  /* check parameters */
  if( reg == NULL ) error("region2rect: invalid region.");
  if( reg_size <= 1 ) error("region2rect: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("region2rect: invalid image 'modgrad'.");
  if( rec == NULL ) error("region2rect: invalid 'rec'.");

  /* center of the region:

     It is computed as the weighted sum of the coordinates
     of all the pixels in the region. The norm of the gradient
     is used as the weight of a pixel. The sum is as follows:
       cx = \sum_i G(i).x_i
       cy = \sum_i G(i).y_i
     where G(i) is the norm of the gradient of pixel i
     and x_i,y_i are its coordinates.
   */
  //获得质心 x,y
  x = y = sum = 0.0;
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].x + reg[i].y * modgrad->xsize ];
      x += (double) reg[i].x * weight;
      y += (double) reg[i].y * weight;
      sum += weight;
    }
  if( sum <= 0.0 ) error("region2rect: weights sum equal to zero.");
  x /= sum;
  y /= sum;

  /* theta */
  //运用惯性矩阵获得更为精确的角度估计
  theta = get_theta(reg,reg_size,x,y,modgrad,reg_angle,prec);
  dx = cos(theta);
  dy = sin(theta);

  /* length and width:

     'l' and 'w' are computed as the distance from the center of the
     region to pixel i, projected along the rectangle axis (dx,dy) and
     to the orthogonal axis (-dy,dx), respectively.

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
   */
  //因为区域的方向向量为 (dx,dy) 
  /*
  ------------------->x
  |\
  | \  
  |  \(dx,dy)
  |   
 \|/
  y
  因此顺时针旋转90°是 (-dy,dx)
  */
  l_min = l_max = w_min = w_max = 0.0;
  for(i=0; i<reg_size; i++)//用向量内积求在线段方向和与线段方向垂直方向的投影求l,w
    {
      l =  ( (double) reg[i].x - x) * dx + ( (double) reg[i].y - y) * dy;
      w = -( (double) reg[i].x - x) * dy + ( (double) reg[i].y - y) * dx;

      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }

  /* store values */
  rec->x1 = x + l_min * dx;
  rec->y1 = y + l_min * dy;
  rec->x2 = x + l_max * dx;
  rec->y2 = y + l_max * dy;
  rec->width = w_max - w_min;
  rec->x = x;
  rec->y = y;
  rec->theta = theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width < 1.0 ) 
	  rec->width = 1.0;
}

//区域质心和角度已经计算好了，因此只进行矩形近似。而region2rect此外还进行了质心和角度计算。
static void region2rect2(point2i * reg, int reg_size,double reg_center_x,double reg_center_y,
					double reg_theta,double prec, double p, struct rect * rec )
{
  double dx,dy,l,w,l_min,l_max,w_min,w_max;
  int i;
  /* check parameters */
  if( reg == NULL ) error("region2rect: invalid region.");
  if( reg_size <= 1 ) error("region2rect: region size <= 1.");
  if( rec == NULL ) error("region2rect: invalid 'rec'.");

  //获得区域的方向向量(dx,dy)
  dx = cos(reg_theta);
  dy = sin(reg_theta);
  l_min = l_max = w_min = w_max = 0.0;
  for(i=0; i<reg_size; i++)//用向量内积求在线段方向和与线段方向垂直方向的投影求l,w
    {
      l =  ( (double) reg[i].x - reg_center_x) * dx + ( (double) reg[i].y - reg_center_y) * dy;
      w = -( (double) reg[i].x - reg_center_x) * dy + ( (double) reg[i].y - reg_center_y) * dx;

      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }

  /* store values */
  rec->x1 = reg_center_x + l_min * dx;
  rec->y1 = reg_center_y + l_min * dy;
  rec->x2 = reg_center_x + l_max * dx;
  rec->y2 = reg_center_y + l_max * dy;
  rec->width = w_max - w_min;
  rec->x = reg_center_x;
  rec->y = reg_center_y;
  rec->theta = reg_theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width < 1.0 ) 
	 rec->width = 1.0;
}
/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point2i (x,y).
 */
static void region_grow( int x, int y, image_double angles, struct point2i * reg,
                         int * reg_size, double * reg_angle, image_char used,
                         double prec )
{
  double sumdx,sumdy;
  int xx,yy,i; 

  /* check parameters */
  if( x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize )
    error("region_grow: (x,y) out of the image.");
  if( angles == NULL || angles->data == NULL )
    error("region_grow: invalid image 'angles'.");
  if( reg == NULL ) error("region_grow: invalid 'reg'.");
  if( reg_size == NULL ) error("region_grow: invalid point2ier 'reg_size'.");
  if( reg_angle == NULL ) error("region_grow: invalid point2ier 'reg_angle'.");
  if( used == NULL || used->data == NULL )
    error("region_grow: invalid image 'used'.");

  /* first point2i of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  *reg_angle = angles->data[x+y*angles->xsize];  /* region's angle */
  sumdx = cos(*reg_angle);
  sumdy = sin(*reg_angle);
  used->data[x+y*used->xsize] = USED;

  /* try neighbors as new region point2is */
  for(i=0; i<*reg_size; i++)
    for(xx=reg[i].x-1; xx<=reg[i].x+1; xx++)
      for(yy=reg[i].y-1; yy<=reg[i].y+1; yy++)
        if( xx>=0 && yy>=0 && xx<(int)used->xsize && yy<(int)used->ysize &&
            used->data[xx+yy*used->xsize] != USED &&
            isaligned(xx,yy,angles,*reg_angle,prec) )
          {
            /* add point2i */
            used->data[xx+yy*used->xsize] = USED;
            reg[*reg_size].x = xx;
            reg[*reg_size].y = yy;
            ++(*reg_size);

            /* update region's angle */
            sumdx += cos( angles->data[xx+yy*angles->xsize] );
            sumdy += sin( angles->data[xx+yy*angles->xsize] );
            *reg_angle = atan2(sumdy,sumdx);
          }
}

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).
 */
static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps )
{
  struct rect r;
  double log_nfa,log_nfa_new;
  double delta = 0.5;
  double delta_2 = delta / 2.0;
  int n;

  log_nfa = rect_nfa(rec,angles,logNT);

  if( log_nfa > log_eps ) return log_nfa;

  /* try finer precisions */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      log_nfa_new = rect_nfa(&r,angles,logNT);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect_copy(&r,rec);
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce width */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.width -= delta;
          log_nfa_new = rect_nfa(&r,angles,logNT);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 += -r.dy * delta_2;
          r.y1 +=  r.dx * delta_2;
          r.x2 += -r.dy * delta_2;
          r.y2 +=  r.dx * delta_2;
          r.width -= delta;
          log_nfa_new = rect_nfa(&r,angles,logNT);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 -= -r.dy * delta_2;
          r.y1 -=  r.dx * delta_2;
          r.x2 -= -r.dy * delta_2;
          r.y2 -=  r.dx * delta_2;
          r.width -= delta;
          log_nfa_new = rect_nfa(&r,angles,logNT);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try even finer precisions */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      log_nfa_new = rect_nfa(&r,angles,logNT);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect_copy(&r,rec);
        }
    }

  return log_nfa;
}

/*----------------------------------------------------------------------------*/
/** Reduce the region size, by elimination the point2is far from the
    starting point2i, until that leads to rectangle with the right
    density of region point2is or to discard the region if too small.
 */
static int reduce_region_radius( struct point2i * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th )
{
  double density,rad1,rad2,rad,xc,yc;
  int i;

  /* check parameters */
  if( reg == NULL ) error("reduce_region_radius: invalid point2ier 'reg'.");
  if( reg_size == NULL )
    error("reduce_region_radius: invalid point2ier 'reg_size'.");
  if( prec < 0.0 ) error("reduce_region_radius: 'prec' must be positive.");
  if( rec == NULL ) error("reduce_region_radius: invalid point2ier 'rec'.");
  if( used == NULL || used->data == NULL )
    error("reduce_region_radius: invalid image 'used'.");
  if( angles == NULL || angles->data == NULL )
    error("reduce_region_radius: invalid image 'angles'.");

  /* compute region point2is density */ //该密度判断已经在函数外判断过，应该可以不用在判断了吧
  density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  // if the density criterion is satisfied there is nothing to do 
  if( density >= density_th ) return TRUE;
  

  /* compute region's radius */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  rad1 = dist( xc, yc, rec->x1, rec->y1 );
  rad2 = dist( xc, yc, rec->x2, rec->y2 );
  rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while( density < density_th )
    {
      rad *= 0.75; /* reduce region's radius to 75% of its value */

      /* remove point2is from the region and update 'used' map */
      for(i=0; i<*reg_size; i++)
        if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) > rad )
          {
            /* point2i not kept, mark it as NOTUSED */
            used->data[ reg[i].x + reg[i].y * used->xsize ] = NOTUSED;
            /* remove point2i from the region */
            reg[i].x = reg[*reg_size-1].x; /* if i==*reg_size-1 copy itself */
            reg[i].y = reg[*reg_size-1].y;
            --(*reg_size);
            --i; /* to avoid skipping one point2i */
          }

      /* reject if the region is too small.
         2 is the minimal region size for 'region2rect' to work. */
      if( *reg_size < 2 ) return FALSE;

      /* re-compute rectangle */
      region2rect(reg,*reg_size,modgrad,reg_angle,prec,p,rec);

      /* re-compute region point2is density */
      density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );
    }

  /* if this point2i is reached, the density criterion is satisfied */
  return TRUE;
}

/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

    For that, an estimation of the angle tolerance is performed by the
    standard deviation of the angle at point2is near the region's
    starting point2i. Then, a new region is grown starting from the same
    point2i, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region point2is,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
static int refine( struct point2i * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th )
{
  double angle,ang_d,mean_angle,tau,density,xc,yc,ang_c,sum,s_sum;
  int i,n;

  /* check parameters */
  if( reg == NULL ) error("refine: invalid point2ier 'reg'.");
  if( reg_size == NULL ) error("refine: invalid point2ier 'reg_size'.");
  if( prec < 0.0 ) error("refine: 'prec' must be positive.");
  if( rec == NULL ) error("refine: invalid point2ier 'rec'.");
  if( used == NULL || used->data == NULL )
    error("refine: invalid image 'used'.");
  if( angles == NULL || angles->data == NULL )
    error("refine: invalid image 'angles'.");

  /* compute region point2is density */
  density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /*------ First try: reduce angle tolerance ------*/

  /* compute the new mean angle and tolerance */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  ang_c = angles->data[ reg[0].x + reg[0].y * angles->xsize ];
  sum = s_sum = 0.0;
  n = 0;
  for(i=0; i<*reg_size; i++)
    {
      used->data[ reg[i].x + reg[i].y * used->xsize ] = NOTUSED;
      if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) < rec->width )
        {
          angle = angles->data[ reg[i].x + reg[i].y * angles->xsize ];
          ang_d = angle_diff_signed(angle,ang_c);
          sum += ang_d;//加上角度差
          s_sum += ang_d * ang_d;//加上角度差的平方
          ++n;
        }
    }
  mean_angle = sum / (double) n;
  //以2倍标准差作为新的角度容忍度，最开始为22.5°*pi/180
  tau = 2.0 * sqrt( (s_sum - 2.0 * mean_angle * sum) / (double) n  +  mean_angle*mean_angle ); /* 2 * standard deviation */
  //以新的角度容忍度重新进行区域生长
  /* find a new region from the same starting point2i and new angle tolerance */
  region_grow(reg[0].x,reg[0].y,angles,reg,reg_size,&reg_angle,used,tau);

  /* if the region is too small, reject */
  if( *reg_size < 2 ) return FALSE;

  /* re-compute rectangle */
  region2rect(reg,*reg_size,modgrad,reg_angle,prec,p,rec);

  /* re-compute region point2is density */
  density = (double) *reg_size /
                      ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /*------ Second try: reduce region radius ------*/
  if( density < density_th )
    return reduce_region_radius( reg, reg_size, modgrad, reg_angle, prec, p,
                                 rec, used, angles, density_th );

  /* if this point2i is reached, the density criterion is satisfied */
  return TRUE;
}
//--------------------------------------------------------
//my code
bool isArcSegment(point2i * reg, int reg_size, struct rect * main_rect, image_double ll_angles,image_char used,image_char pol,
                         double prec, double p, rect * rect_up, rect * rect_down)
{
	point2i * reg_up = (point2i*)malloc(reg_size*sizeof(point2i));
	point2i * reg_down = (point2i*)malloc(reg_size*sizeof(point2i));
	int   reg_up_size,reg_down_size;
	double reg_up_theta,reg_down_theta, main_theta;
	double reg_up_sin_s,reg_up_cos_s,reg_down_sin_s,reg_down_cos_s;
	double reg_up_x,reg_up_y,reg_down_x,reg_down_y;
	//double weight,sum;
	double temp1,temp2;
	int same_pol_cnt,opp_pol_cnt;
	int i;

	same_pol_cnt = opp_pol_cnt = 0;
	reg_up_size = reg_down_size = 0;

	for ( i = 0; i < reg_size; i++)
	{
		switch(pol->data[reg[i].y*pol->xsize+reg[i].x])
		{
			case SAME_POL: same_pol_cnt++;break;//统计同极性的pixel数量
			case OPP_POL : opp_pol_cnt++; break;//统计反极性的pixel数量
			default:break;
		}
	 //选与theta角度为法线方向，过质心的直线方程为 dx*(x-xi)+dy*(y-yi)=0,则与方向相同的点代入方程得到距离d,d>=0归入reg_up,d<0归入reg_down
	  if( main_rect->dx*( reg[i].x - main_rect->x ) + main_rect->dy*( reg[i].y - main_rect->y ) >= 0)
		  reg_up[reg_up_size++] = reg[i];
	  else
		  reg_down[reg_down_size++] = reg[i];
	}
	//对于已经被标记过极性的区域，我们没必要再进行极性分析
	if( (same_pol_cnt + opp_pol_cnt) > reg_size/2)
	{
		if(same_pol_cnt > opp_pol_cnt )
		{
			main_rect->polarity = 1;
		    rect_up->polarity = 1;
	        rect_down->polarity = 1;
		}
		else
		{
			main_rect->polarity = -1;
		    rect_up->polarity = -1;
	        rect_down->polarity = -1;
		}
		return TRUE;
	}
	//计算与主方向相同的上半部分区域质心
	reg_up_x = reg_up_y = 0;
	//sum = 0;
	reg_up_sin_s = reg_up_cos_s = 0;
	for ( i = 0; i< reg_up_size; i++)
	{
		//weight = modgrad->data[ reg_up[i].x + reg_up[i].y * modgrad->xsize ];
		//reg_up_x += (double)weight*reg_up[i].x;
		//reg_up_y += (double)weight*reg_up[i].y;
		//sum += weight;
		reg_up_sin_s += sin(ll_angles->data[ reg_up[i].x + reg_up[i].y * ll_angles->xsize ]);
		reg_up_cos_s += cos(ll_angles->data[ reg_up[i].x + reg_up[i].y * ll_angles->xsize ]);
	}
	//reg_up_x /= sum;
	//reg_up_y /= sum;
	reg_up_theta = atan2(reg_up_sin_s,reg_up_cos_s);
	//计算主方向上的下半部分区域质心
	reg_down_x = reg_down_y = 0;
	//sum = 0;
	reg_down_sin_s = reg_down_cos_s = 0;
	for ( i = 0; i< reg_down_size; i++)
	{
		//weight = modgrad->data[ reg_down[i].x + reg_down[i].y * modgrad->xsize ];
		//reg_down_x += (double)weight*reg_down[i].x;
		//reg_down_y += (double)weight*reg_down[i].y;
		//sum += weight;
		reg_down_sin_s += sin(ll_angles->data[ reg_down[i].x + reg_down[i].y * ll_angles->xsize ]);
		reg_down_cos_s += cos(ll_angles->data[ reg_down[i].x + reg_down[i].y * ll_angles->xsize ]);
	}
	//reg_down_x /= sum;
	//reg_down_y /= sum;
	reg_down_theta = atan2(reg_down_sin_s,reg_down_cos_s);
	main_theta  = atan2(reg_up_sin_s+reg_down_sin_s,reg_up_cos_s+reg_down_cos_s);
	//估计两个区域方向
	//reg_up_theta = get_theta(reg_up,reg_up_size,reg_up_x,reg_up_y,modgrad,main_rect->theta,prec);
	//reg_down_theta = get_theta(reg_down,reg_down_size,reg_down_x,reg_down_y,modgrad,main_rect->theta,prec);
	//旋转到0°进行比较theta,reg_up_theta,reg_down_theta
	temp1 = angle_diff_signed(reg_up_theta,main_theta);
	temp2 = angle_diff_signed(reg_down_theta,main_theta);
	/*if(temp1>= M_PI/2 || temp1 <= -M_PI/2)
		temp1 += 0;
	if(temp2>= M_PI/2 || temp2 <= -M_PI/2)
		temp2 += 0;*/
	//if(temp1 >= prec/10 && temp2 <= -prec/10)//顺时针,边缘的梯度方向与弧的指向圆心方向相反，polarity = -1
	if(temp1 >= M_1_8_PI/10 && temp2 <= -M_1_8_PI/10)//实验证明取定值效果更好
	{
		main_rect->polarity = -1;
		rect_up->polarity = -1;
	    rect_down->polarity = -1;
		//标记极性
	    for ( i = 0; i < reg_size; i++)
	    {
			pol->data[reg[i].y*pol->xsize+reg[i].x] = OPP_POL;//-1
	    }
	}
	//else if(temp1 <= -prec/10 && temp2 >= prec/10)//逆时针，边缘的梯度方向与弧的指向圆心方向相同，polarity = 1
	else if(temp1 <= -M_1_8_PI/10 && temp2 >= M_1_8_PI/10)//实验证明取定值效果更好
	{
		main_rect->polarity = 1;
		rect_up->polarity = 1;
	    rect_down->polarity = 1;
		//标记极性
	    for ( i = 0; i < reg_size; i++)
	    {
			pol->data[reg[i].y*pol->xsize+reg[i].x] = SAME_POL;//1
	    }
	}
	else
	{
		//在region_grow中已经置为USED了
		//for ( i = 0; i< reg_size; i++)
		//	used->data[reg[i].y*used->xsize+reg[i].x] = USED;
		return FALSE;
	}
	
	//region2rect2(reg_up,reg_up_size,reg_up_x,reg_up_y,reg_up_theta,prec,p,rect_up);
	//region2rect2(reg_down,reg_down_size,reg_down_x,reg_down_y,reg_down_theta,prec,p,rect_down);

	free(reg_up);
	free(reg_down);
	return TRUE;
}

/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD full interface.
 */
double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y )
{
  image_double image;
  ntuple_list out = new_ntuple_list(8);
  double * return_value;
  image_double scaled_image,angles,modgrad;
  image_char used;
  image_char pol;  //对于构成圆弧的像素标记极性，如果梯度的方向和弧的方向指向一致，则为SAME_POLE,否则为OPP_POLE,该标记初始是为0
  image_int region = NULL;
  struct coorlist * list_p;
  struct coorlist * list_p_temp;
//  struct coorlist * mem_p;
  struct rect main_rect;//main rect
  struct rect rect_up,rect_down;//divide the rect into 2 rects:rect_up and rect_down
  struct point2i * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize;
  double rho,reg_angle,prec,p;
  double log_nfa = -1,logNT;
//  double log_nfa1,log_nfa2;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */
  int seed_cnt = 0;
  int refine_cnt = 0;
  int reg_size_toosmall_cnt=0;

  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 ) error("invalid image input.");
  if( scale <= 0.0 ) error("'scale' value must be positive.");
  if( sigma_scale <= 0.0 ) error("'sigma_scale' value must be positive.");
  if( quant < 0.0 ) error("'quant' value must be positive.");
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");


  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;

  rho = quant / sin(prec); /* gradient magnitude threshold */


  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr( (unsigned int) X, (unsigned int) Y, img );
  if( scale != 1.0 )
    {
	  //按照scale进行高斯降采样的图像，注意宽高是上取整，设采样后高宽为imgN*imgM
      scaled_image = gaussian_sampler( image, scale, sigma_scale );
	  //返回一张梯度角度顺时针旋转90°后的align角度图angles，如果梯度角度是(gx,gy)->(-gy,gx)，
	  //和梯度的模的图modgrad,然后按照n_bins进行伪排序返回链表的头指针list_p,里面存的是坐标
	  angles = ll_angle( scaled_image, rho, &list_p,&modgrad, (unsigned int) n_bins );
      free_image_double(scaled_image);
    }
  else
    angles = ll_angle( image, rho, &list_p,&modgrad,(unsigned int) n_bins );
  xsize = angles->xsize;//降采样后的图像的x size，宽度imgM
  ysize = angles->ysize;//降采样后的图像的y size，高度imgN

  /* Number of Tests - NT

     The theoretical number of tests is Np.(XY)^(5/2)
     where X and Y are number of columns and rows of the image.
     Np corresponds to the number of angle precisions considered.
     As the procedure 'rect_improve' tests 5 times to halve the
     angle precision, and 5 more times after improving other factors,
     11 different precision values are potentially tested. Thus,
     the number of tests is
       11 * (X*Y)^(5/2)
     whose logarithm value is
       log10(11) + 5/2 * (log10(X) + log10(Y)).
  */
  logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0
          + log10(11.0);
  min_reg_size = (int) (-logNT/log10(p)); /* minimal number of point2is in region that can give a meaningful event，每个矩形区域内align point2i最小数量*/
  /* initialize some structures */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL ) /* save region data */
    region = new_image_int_ini(angles->xsize,angles->ysize,0);//申请与降采样后图像一样大小的int类型的内存，该内存的作用是将检测到的线段序号标到相应的图像格子里，该部分可有可无
  used = new_image_char_ini(xsize,ysize,NOTUSED);//申请与降采样后图像一样大小的char类型的内存
  pol  = new_image_char_ini(xsize,ysize,NOTDEF_POL);//像素点处的梯度和弧指向的方向的极性标记
  reg = (struct point2i *) calloc( (size_t) (xsize*ysize), sizeof(struct point2i) );
  if( reg == NULL ) error("not enough memory!");

  list_p_temp = list_p;//记录头链表的头指针，后面需要利用该头指针进行内存释放
  /* search for line segments */
  for(; list_p_temp != NULL; list_p_temp = list_p_temp->next )
    if( used->data[ list_p_temp->x + list_p_temp->y * used->xsize ] == NOTUSED &&
        angles->data[ list_p_temp->x + list_p_temp->y * angles->xsize ] != NOTDEF )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
      {
        /* find the region of connected point2i and ~equal angle */
		//reg是长度为imgN*imgM的一维point2i型数组，有足够大的空间存储生长的区域，reg_size是里面存储了数据的数量，记录的是区域的point2i
		//reg_angle是该区域的主方向的double型变量，存的角度是弧度制
		  seed_cnt ++;
        region_grow( list_p_temp->x, list_p_temp->y, angles, reg, &reg_size,&reg_angle, used, prec );

        /* reject small regions */
        if( reg_size < min_reg_size ) 
		{
			reg_size_toosmall_cnt++;
			continue;
		}

        /* construct rectangular approximation for the region */
		//根据生长的区域得到近似外接矩阵的参数，矩形参数包括:起点，终点，方向theta，宽度等
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&main_rect);
		if( FALSE == isArcSegment(reg,reg_size,&main_rect,angles,used,pol,prec,p,&rect_up,&rect_down))
			continue;
        /* Check if the rectangle exceeds the minimal density of
           region point2is. If not, try to improve the region.
           The rectangle will be rejected if the final one does
           not fulfill the minimal density condition.
           This is an addition to the original LSD algorithm published in
           "LSD: A Fast Line Segment Detector with a False Detection Control"
           by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
           The original algorithm is obtained with density_th = 0.0.
         */

        //提纯，通过重新生长区域来达到期望的密度阈值 
        if( !refine( reg, &reg_size, modgrad, reg_angle,
                     prec, p, &main_rect, used, angles, density_th ) ) continue;

		refine_cnt++;
        // compute NFA value 
        log_nfa = rect_improve(&main_rect,angles,logNT,log_eps);//通过改善矩形区域以尝试得到期望的nfa值
        if( log_nfa <= log_eps ) //错误控制
			continue;
        // A New Line Segment was found! 
        ++ls_count;  // increase line segment counter 

        //
        //  The gradient was computed with a 2x2 mask, its value corresponds to
        //  point2is with an offset of (0.5,0.5), that should be added to output.
        //  The coordinates origin is at the center of pixel (0,0).
        //
        main_rect.x1 += 0.5; main_rect.y1 += 0.5;
        main_rect.x2 += 0.5; main_rect.y2 += 0.5;

        // scale the result values if a subsampling was performed */
        if( scale != 1.0 )
          {
            main_rect.x1 /= scale; main_rect.y1 /= scale;
            main_rect.x2 /= scale; main_rect.y2 /= scale;
          //  main_rect.width /= scale;
          }

        /* add line segment found to output */
		add_8tuple( out, main_rect.x1, main_rect.y1, main_rect.x2, main_rect.y2,main_rect.dx,main_rect.dy,
			dist(main_rect.x1, main_rect.y1, main_rect.x2, main_rect.y2), main_rect.polarity);

		//------------------------------------------------------------------------------------------------- 
		/*
		cout<<ls_count<<'\t'<<main_rect.theta<<'\t'<<main_rect.theta*180/M_PI<<"\t polarity:"<<main_rect.polarity<<endl;//打印theta
		
			fstream file1,file2;
			if(ls_count == 1)//清空内容
			{
				file1.open("D:\\Graduate Design\\picture\\sp\\coor.txt",ios::out | ios::trunc);
				file1.close();
				file2.open("D:\\Graduate Design\\picture\\sp\\reg.txt",ios::out | ios::trunc);
				file2.close();
			}
			
			file1.open("D:\\Graduate Design\\picture\\sp\\coor.txt",ios::app);
			file1<<main_rect.x1<<'\t'<<main_rect.y1<<'\t'<<main_rect.x2<<'\t'<<main_rect.y2<<'\t'<<(main_rect.theta*180/M_PI)<<endl;
			file1.close();
			
			if(ls_count == 1)//保持第1根线段的区域
			{
				file2.open("D:\\Graduate Design\\picture\\sp\\reg.txt",ios::app);
				for(i=0; i<reg_size; i++)
					file2<<angles->data[ reg[i].x + reg[i].y * angles->xsize ]*180/M_PI<<endl;
				file2.close();
			}
			*/
		//-------------------------------------------------------------------------------------------------------
        /* add region number to 'region' image if needed */ //将检测到的线段序号标到相应的图像格子里，该部分可有可无
        if( region != NULL )
          for(i=0; i<reg_size; i++)
            region->data[ reg[i].x + reg[i].y * region->xsize ] = ls_count;
      }


  /* free memory */
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data point2ier was provided to this functions
                               and should not be destroyed.                 */
  free_image_double(angles);
  free_image_double(modgrad);
  free_image_char(used);
  free_image_char(pol);
  free( (void *) reg );
//  free( (void *) mem_p );
  //释放分成1024区的存储梯度从大到小的链表,mycode
  //---------------------------------------
  list_p_temp = list_p->next;
  while(list_p_temp != NULL)
  {
	  free(list_p);
	  list_p = list_p_temp;
	  list_p_temp = list_p->next;
  }
  free(list_p);

  //cout<<"seed cnt:"<<seed_cnt<<endl;
  //cout<<"refine cnt:"<<refine_cnt<<endl;
  //cout<<"reg_size_toosmall cnt:"<<reg_size_toosmall_cnt<<endl;
  //----------------------------------------
  /* return the result */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL )
    {
      if( region == NULL ) error("'region' should be a valid image.");
      *reg_img = region->data;
      if( region->xsize > (unsigned int) INT_MAX ||
          region->xsize > (unsigned int) INT_MAX )
        error("region image to big to fit in INT sizes.");
      *reg_x = (int) (region->xsize);
      *reg_y = (int) (region->ysize);

      /* free the 'region' structure.
         we cannot use the function 'free_image_int' because we need to keep
         the memory with the image data to be returned by this function. */
      free( (void *) region );
    }
  if( out->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");
  *n_out = (int) (out->size);

  return_value = out->values;
  free( (void *) out );  /* only the 'ntuple_list' structure must be freed,
                            but the 'values' point2ier must be keep to return
                            as a result. */
  return return_value;
}

/*------------------------------------------------------------------------------------------------*/
/**
my code,Alan Lu
输入
img  : 输入图像的一维double型数组,大小为Y*X，按照行优先存储，传入前需要拥有内存
X    : 输入图像的columns
Y    ：输入图像的rows
输出
n_out: lsd算法检测得到的线段的数量n，return的返回值是n条线段，为一维double型数组，长度为8*n，每8个为一组，存着x1,y1,x2,y2,dx,dy,width,polarity
reg_img: 输出标记区域，是一维的int型数组，大小reg_y*reg_x,在相应的像素位置标记着它属于的线段(1,2,3,...n),如果值为0表示不属于任何线段.
         假如外部是int * region_img,则只需要 &region_img,就可以得到标记区域的返回，不需要时直接NULL传入
reg_x  : 输出标记区域的columns,不需要时直接NULL传入
reg_y  : 输出标记区域的rows,不需要时直接NULL传入
*/
double * mylsd(int * n_out, double * img, int X, int Y, int ** reg_img, int * reg_x, int * reg_y)
{
	 /* LSD parameters */
  double scale = 0.8;       /* Scale the image by Gaussian filter to 'scale'. */
  double sigma_scale = 0.6; /* Sigma for Gaussian filter is computed as
                                sigma = sigma_scale/scale.                    */
  double quant = 2.0;       /* Bound to the quantization error on the
                                gradient norm.                                */
  double ang_th = 22.5;     /* Gradient angle tolerance in degrees.           */
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  double density_th = 0.7;  /* Minimal density of region point2is in rectangle. */
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */ 

  return LineSegmentDetection( n_out, img, X, Y, scale, sigma_scale, quant,
                               ang_th, log_eps, density_th, n_bins,
                               reg_img, reg_x, reg_y );
}
//lines: 输入的lines_num条线段，每条线段8个值，存着x1,y1,x2,y2,dx,dy,width,polarity
//lines_num:
//new_lines_num: 拒绝短线段后的new_lines_num条线段，存在lines的前面，而短的线段会放到尾巴处
//此处长度限制参数很重要：目前取8^2, 14^2
void     rejectShortLines(double * lines, int lines_num, int * new_lines_num )
{
	int    new_num = 0;
	int    shor_lines_num = 0;
	double temp;
	new_num = lines_num - shor_lines_num;
	for ( int i = 0; i< new_num; i++)
	{
		if( lines[i*8+6] < 10)//reject short lines, the length threshold is important: 8,14 最后需要调节
		{
			for ( int j = 0; j<8; j++)
			{
				temp = lines[i*8+j];
				lines[i*8+j] = lines[(new_num-1)*8+j];
				lines[(new_num-1)*8+j] = temp;
			}
			i--; //调换后需要检查调换来的线段长度，需要回退
			shor_lines_num++;
			new_num = lines_num - shor_lines_num;
		}
	}
	*new_lines_num = new_num;
}

/*----------------------------------------------------------------------------*/
//输入：
//start_angle,end_angle, 角度方位是(-pi,pi).  
//  pi    ------->x  0
//        |
//        |
//       y\/ pi/2
//polarity: 当polarity为1时，表示的是从start_angle按照逆时针方向旋转到end_angle的角度;当polarity为-1时，表示的是从start_angle按照顺时针方向旋转到end_angle的角度;
//返回值： 旋转角度coverage
inline double rotateAngle(double start_angle, double end_angle, int polarity)
{
	double coverage;
	//首先需要将angle1和angle2转换到 0 ~ 2pi
	if(start_angle < 0) start_angle += M_2__PI;//限制角度在0~2pi之间
	if(end_angle < 0 ) end_angle += M_2__PI;
	if(polarity == 1)//极性为1
	{
		coverage = start_angle - end_angle;
	}
	else //极性为-1
	{ 
		coverage = end_angle - start_angle;
	}
	if(coverage < 0) coverage += M_2__PI;
	return coverage;
}
//对线段按照凸性和距离进行分组
//lines: 输入的lines_num条线段，每条线段8个值，存着x1,y1,x2,y2,dx,dy,length,polarity
//lines_num:
//输出分组groups. 每个组是一个vector<int>
//注意：切记用完region,需要在函数外面手动释放region
void groupLSs(double *lines, int line_num, int * region, int imgx, int imgy, vector<vector<int>> * groups)
{
	if(line_num == 0)
	{
		groups = NULL;
		return;
	}
	unsigned char isEnd = 0;//是否还可以继续搜寻
	int currentLine; //当前线段
	char * label = (char*)calloc(line_num, sizeof(char));
	memset(label,0,sizeof(char)*line_num); //init the label all to be zero
	int * group_up = (int*)malloc(sizeof(int)*line_num);//申请足够内存，存储延线段方向得到的分组的线段
	int * group_down = (int*)malloc(sizeof(int)*line_num);//存储线段反方向分组的线段
	int group_up_cnt,group_down_cnt;
	//coorlist * head,*tail;
	vector<int> group_temp;
	point2d dir_vec1,dir_vec2;
	point2i *votebin = (point2i*)calloc(line_num,sizeof(point2i));//申请足够内存，用来投票. x记录线段索引，y记录票数
	int bincnt = 0;
	int xx,yy,temp;
	double start_angle,end_angle,angle_delta;
	for ( int i = 0; i<line_num; i++)
	{
		if( label[i] == 0)//未被分组过
		{
			group_up_cnt = group_down_cnt = 0;//每开始寻找一组，需要置零
			//先从第i条线段的头部开始搜索，进行分组,结果存在group_up里面
			group_up[group_up_cnt++] = i;//记录线段i,注意线段是0~line_num-1
			isEnd = 0;//置零，表示还可以从当前线段开始搜索，还未结束
	     	currentLine = i;
			while(isEnd == 0)
			{
				label[currentLine] = 1; //标记该线段已经被分组
				//head = tail = NULL;
		        bincnt = 0;
				dir_vec1.x = lines[currentLine*8+4];
				dir_vec1.y = lines[currentLine*8+5];
				if ( lines[currentLine*8+7] == 1)//极性为正
				{
					//将dir_vec1逆时针旋转45°
					dir_vec2.x = (dir_vec1.x + dir_vec1.y)*0.707106781186548; // sqrt(2)/2 = 0.707106781186548
				    dir_vec2.y = (-dir_vec1.x + dir_vec1.y)*0.707106781186548;
				}
				else
				{
					//将dir_vec1顺时针旋转45°
					dir_vec2.x = (dir_vec1.x - dir_vec1.y)*0.707106781186548; // sqrt(2)/2 = 0.707106781186548
				    dir_vec2.y = (dir_vec1.x + dir_vec1.y)*0.707106781186548;
				}
				for ( int j = 1; j<=4; j++)
					for ( int k = 1; k<=4; k++)//在4x4邻域内搜索
					{
						xx = (int)(lines[currentLine*8+2]*0.8+j*dir_vec1.x+k*dir_vec2.x);
						yy = (int)(lines[currentLine*8+3]*0.8+j*dir_vec1.y+k*dir_vec2.y);
						if(xx < 0 || xx >= imgx || yy < 0 || yy >= imgy)//越界
							continue;
						temp = region[yy*imgx+xx];
						if(temp>0)//表示有线段的支持区域，在1~line_num
						{
							region[yy*imgx+xx] = -temp;//取负数标记
							for (xx = 0; xx<bincnt; xx++)
							{
								if(votebin[xx].x == temp - 1)//如果以前投票过，直接在相应的bin的票数上加1
								{
									votebin[xx].y++;
									break;
								}
							}
							if(xx == bincnt)//如果以前没有投票过，增加该线段，并记录票数为1
							{
								if(bincnt == line_num)
									error("group ls error1");
								votebin[bincnt].x = temp - 1;
								votebin[bincnt].y = 1;
								bincnt++; //bin的总数加1
							}
						}
					}
			    //寻找投票最多的线段，并且需要满足数量大于一定值
			    temp = 0;
				for ( int j = 0; j<bincnt; j++)
				{
					if(votebin[j].y>temp)
					{
						temp = votebin[j].y;
						xx = votebin[j].x;//借用xx变量
					}
				}
				if ( temp >= 5 && label[xx] == 0 && lines[8*xx+7] == lines[8*i+7] )//待实验调整参数值
				{
					if(group_up_cnt == line_num)
					   error("group ls error2");
					yy = group_up_cnt-1;//借用yy变量
					start_angle = atan2(lines[8*group_up[yy]+5],lines[8*group_up[yy]+4]);
					end_angle = atan2(lines[8*xx+5],lines[8*xx+4]);
					angle_delta = rotateAngle(start_angle,end_angle,(int)lines[8*i+7]);
					if(angle_delta <= M_3_8_PI)//相邻两线段的旋转夹角也需要满足在pi/4内
					{
						group_up[group_up_cnt++] = xx;//压入线段
						currentLine = xx; //更新当前搜索线段
					}
					else
						isEnd = 1;
				}
				else
					isEnd = 1;//结束，已经找不到可以分组的线段了
			}
			//先从第i条线段的尾部开始搜索，进行分组,结果存在group_down里面。记住，第i条线段在group_up和group_down中的0索引处都储存了
			group_down[group_down_cnt++] = i; 
			isEnd = 0;//置零，表示还可以从当前线段开始搜索，还未结束
	     	currentLine = i;
			while(isEnd == 0)
			{
				label[currentLine] = 1; //标记该线段已经被分组
				//head = tail = NULL;
		        bincnt = 0;
				dir_vec1.x = -lines[currentLine*8+4];
				dir_vec1.y = -lines[currentLine*8+5];
				if ( lines[currentLine*8+7] == 1)//极性相同
				{
					//将dir_vec1顺时针旋转45°
					dir_vec2.x = (dir_vec1.x - dir_vec1.y)*0.707106781186548; // sqrt(2)/2 = 0.707106781186548
				    dir_vec2.y = (dir_vec1.x + dir_vec1.y)*0.707106781186548;
				}
				else
				{
					//将dir_vec1顺时针旋转45°
					dir_vec2.x = (dir_vec1.x + dir_vec1.y)*0.707106781186548; // sqrt(2)/2 = 0.707106781186548
				    dir_vec2.y = (-dir_vec1.x + dir_vec1.y)*0.707106781186548;
				}
				for ( int j = 1; j<=4; j++)
					for ( int k = 1; k<=4; k++)//在4x4邻域内搜索
					{
						xx = (int)(lines[currentLine*8+0]*0.8+j*dir_vec1.x+k*dir_vec2.x);
						yy = (int)(lines[currentLine*8+1]*0.8+j*dir_vec1.y+k*dir_vec2.y);
						if(xx < 0 || xx >= imgx || yy < 0 || yy >= imgy)//越界
							continue;
						temp = region[yy*imgx+xx];
						if(temp>0)//表示有线段的支持区域，在1~line_num
						{
							region[yy*imgx+xx] = -temp;//取负数标记
							for (xx = 0; xx<bincnt; xx++)
							{
								if(votebin[xx].x == temp - 1)//如果以前投票过，直接在相应的bin的票数上加1
								{
									votebin[xx].y++;
									break;
								}
							}
							if(xx == bincnt)//如果以前没有投票过，增加该线段，并记录票数为1
							{
								if(bincnt == line_num)
									error("group ls error3");
								votebin[bincnt].x = temp - 1;
								votebin[bincnt].y = 1;
								bincnt++; //bin的总数加1
							}
						}
					}
			    //寻找投票最多的线段，并且需要满足数量大于一定值
			    temp = 0;
				for ( int j = 0; j<bincnt; j++)
				{
					if(votebin[j].y>temp)
					{
						temp = votebin[j].y;
						xx = votebin[j].x;//借用xx变量
					}
				}
				if ( temp >= 5 && label[xx] == 0 && lines[8*xx+7] == lines[8*i+7])//待实验调整参数值
				{
					if(group_down_cnt == line_num)
					   error("group ls error2");
					yy = group_down_cnt-1;//借用yy变量
					start_angle = atan2(lines[8*group_down[yy]+5],lines[8*group_down[yy]+4]);
					end_angle = atan2(lines[8*xx+5],lines[8*xx+4]);
					angle_delta = rotateAngle(end_angle,start_angle,(int)lines[8*i+7]);//注意此时需要调换一下，因为是从尾部开始搜索
					if(angle_delta < M_3_8_PI)//相邻两线段的旋转夹角也需要满足在pi/4内,pi*3/8 = 66.5°
					{
						group_down[group_down_cnt++] = xx; //压入线段
						currentLine = xx; //更新当前搜索线段
					}
					else
						isEnd = 1;
				}
				else
					isEnd = 1;//结束，已经找不到可以分组的线段了
			}
			(*groups).push_back(group_temp); //添加线段分组
			temp = (*groups).size()-1;
			for (int j = group_down_cnt-1; j>= 0; j--)
			{
				(*groups)[temp].push_back(group_down[j]);
			}
			for (int j = 1; j<group_up_cnt; j++)//由于第i条线段在group_up和group_down都储存了，所以就从索引1开始
			{
				(*groups)[temp].push_back(group_up[j]);
			}
		}
	}
	free(label);
	free(group_up);
	free(group_down);
	free(votebin);
}
//计算groups中每个组的跨度
//输入：
//lines: 输入的lines_num条线段，每条线段8个值，存着x1,y1,x2,y2,dx,dy,length,polarity
//lines_num:
//groups: 分组，每个分组都存着线段的索引
//输出:
//coverages: 每个组的跨度，当组内线段只有1条时，跨度为0. coverages的长度等于组的数量 = groups.size()
//注意，coverages用前不需要申请内存，coverages用完后，需要在函数外手动释放内存，长度等于分组数量
void calcuGroupCoverage(double * lines, int line_num, vector<vector<int>> groups, double * &coverages)
{
	int groups_num = groups.size();
	int temp;
	double start_angle,end_angle;
	coverages = (double*)malloc(sizeof(double)*groups_num);
	for ( int i = 0; i<groups_num; i++)
	{
		temp = groups[i].size()-1;
		if(groups[i].size() == 0)//第i个分组只有1条线段，则跨度为0
		{
			coverages[i] = 0;
		}
		else
		{
			start_angle = atan2(lines[8*groups[i][0]+5],lines[8*groups[i][0]+4]);
			end_angle = atan2(lines[8*groups[i][temp]+5],lines[8*groups[i][temp]+4]);
			coverages[i] = rotateAngle(start_angle,end_angle,(int)lines[8*groups[i][0]+7]);
		}
	}
}

//==============================================================================
//====================================================================================================
//================================clustering==========================================================
//聚类
//求points中第i行与initializations中第j行里每个元素的平方差总和,每行维度都为nDims
inline double squaredDifference(int & nDims, double *& points, int & i, double *& initializations, int & j)
{
    double result = 0;
    for (int k = 0; k < nDims; ++k)
		result += pow(points[i*nDims+k] - initializations[j*nDims+k], 2);
    return result;
}
/**
 *输入
 *points: 待均值漂移的点集，总共有nPoints个点，每个点有nDims维度，是一维数组
 *initPoints: 均值漂移初始化位置，在nxd空间中找均值漂移初始时开始搜索的位置，总共有initLength个点，每个点有nDims维度
 *sigma = 1
 *window_size: window parameter = distance_tolerance或者window parameter = distance_tolerance/2
 *accuracy_tolerance: 收敛容忍误差1e-6
 *iter_times: 迭代次数50
 *输出
 *收敛的位置，位置个数与初始化搜索位置个数一样,我们将结果更新到initPoints,也就是它既是输入参数，也是输出参数，节省内存
 */
void meanShift( double * points, int nPoints, int nDims, double * & initPoints, int initLength, double sigma, double window_size, double accuracy_tolerance, int iter_times )
{
//	for (int i = 0; i<initLength; i++)
//		cout<<initPoints[2*i]<<'\t'<<initPoints[2*i+1]<<endl;
    int nQuerries = initLength;
    double * initializations = (double*)malloc(nQuerries * nDims * sizeof(double));
    memcpy(initializations, initPoints , nQuerries * nDims * sizeof(double));//copy

    double sigma2 = sigma*sigma;//sigma平方
    double radius2 = window_size *window_size;//平方
    double tolerance = accuracy_tolerance;
    int iters, maxiters = iter_times;//最大迭代次数
   //返回与初始搜索点集一样大小的最终定位点集
    double * finals = (double*)malloc(nQuerries * nDims * sizeof(double));;//最终定位点集的指针
    memcpy(finals, initializations, nQuerries * nDims * sizeof(double));
	double * distances = (double*)malloc(nPoints*sizeof(double));
    //printf("meanShift: nPoints:%d \tnDims: %d \tnQuerries:%d \n",nPoints,nDims,nQuerries);//打印
    for (int loop = 0; loop < nQuerries; ++loop)
    {
        iters = 0;
        while (iters < maxiters)
        {
            bool flag = false;
            double denominator = 0;//分母
            for (int i = 0; i < nPoints; ++i)//对所有的点集进行遍历，找到落在搜索圆域内的点
            {
                distances[i] = squaredDifference(nDims, points, i, initializations, loop);//求距离的平方
                if (distances[i] <= radius2)//在第loop个搜索中心的以sqrt(radius2)为半径的圆域内
                {
                    flag = true;
                    denominator += exp(-distances[i] / sigma2);
                }
            }
            if (!flag)
                break;
            for (int j = 0; j < nDims; ++j)
				finals[loop*nDims+j] = 0;//对最终定位点集中的第loop个点的向量赋值为0
            for (int i = 0; i < nPoints; ++i)
                if (distances[i] <= radius2)
                {
                    for (int j = 0; j < nDims; ++j)//每个内点向量的以一定权值累加
						finals[loop*nDims+j] += exp(-distances[i] / sigma2) * points[i*nDims+j];
                }
            for (int j = 0; j < nDims; ++j)//权值归一化
				finals[loop*nDims+j] /= denominator;
            if (sqrt(squaredDifference(nDims, finals, loop, initializations, loop)) < tolerance)//相继两次的迭代中心在误差内了，则认为已经收敛，没必要再继续迭代
                break;
            iters = iters + 1;
            for (int j = 0; j < nDims; ++j)//更新迭代的搜索中心
				initializations[loop*nDims+j] = finals[loop*nDims+j];
        }
    }
	memcpy(initPoints, finals, nQuerries * nDims * sizeof(double));
    free(distances);
    free(initializations);
	free(finals);
}

/***
 *输入
 *points,待聚类的点集,为一维数组,nPoints个点，每个点维度是nDims
 *distance_threshold 决定聚类的距离阈值
 *输出 outPoints
 *聚类后的点集 nOutPoints x nDims 
 *该函数要千万注意，当被调用后，函数内部会多申请nOutPoints个double型的数组内存，在外边使用完毕后，切记free(outPoints).
 */
void clusterByDistance(double * points, int nPoints, int nDims, double distance_threshold,int number_control, double * & outPoints, int * nOutPoints)
{ 
	double threshold2 = distance_threshold*distance_threshold;
    std::vector<double*> centers;
    std::vector<int> counts;
    centers.clear();
    counts.clear();
	char * labeled = (char*)malloc(sizeof(char)*nPoints);
    memset(labeled, 0, nPoints * sizeof(char));//初始化bool型标签为0
	if(nPoints == 1)
	{
		centers.push_back((double*)malloc(sizeof(double)*nDims));
		for (int k = 0; k < nDims; ++k)
			centers[centers.size() - 1][k] = points[k];
        counts.push_back(1);
	}
	else
	{
		for (int i = 0; i < nPoints; ++i)
		{
		    if (!labeled[i])
			{
		        labeled[i] = 1;
				centers.push_back((double*)malloc(sizeof(double)*nDims));
		        counts.push_back(1);
		        for (int k = 0; k < nDims; ++k)
				{
				   centers[centers.size() - 1][k] = points[i*nDims+k];  
				}
		        for (int j = i+1; j < nPoints; ++j)
		        {
		            if (!labeled[j])
		            {
		                double d = 0;
		                for (int k = 0; k < nDims; ++k)
				            d += pow(centers[centers.size() - 1][k] / counts[centers.size() - 1] - points[j*nDims+k], 2);
		                if (d <= threshold2)
		                {
		                    ++counts[centers.size() - 1];
		                    for (int k = 0; k < nDims; ++k)
								centers[centers.size() - 1][k] += points[j*nDims+k];
		                    labeled[j] = 1;
							if(counts[centers.size() - 1] >= number_control)//聚类数量控制，防止均值中心漂的太远  圆心聚类时20  半径聚类时10
								break;
		                }
		            }
		        }
		    }
		}
	}
    free(labeled);
    centers.shrink_to_fit();
    counts.shrink_to_fit();
    int m = (int) centers.size();
    outPoints = (double*)malloc(sizeof(double)*m*nDims);
	(*nOutPoints) = m;
    for (unsigned int i = 0; i < centers.size(); ++i)
    {
        for (int j = 0; j < nDims; ++j)
		{
			outPoints[i*nDims+j] = centers[i][j] / counts[i];
//			cout<<out[i*nDims+j]<<'\t';
		}
//		cout<<endl;
        free(centers[i]);
    }
    centers.resize(0);
    counts.resize(0);
//	vector<double*>().swap(centers);//释放回收vector内存
//	vector<int>().swap(counts);
}

//聚类算法，均值漂移
//三个步骤，一是选取初始迭代点，二是均值漂移，三是去除重复点，从而得到聚类中心
//获得候选圆心的聚类中心(xi,yi)
//输入：
//points，一维点数据,长度为points_num x 2
//distance_tolerance,数据点聚类的半径
//输出：
//二维数据点的聚类中心 centers是一维double数组， 大小为 centers_num x 2
//正确返回值为1，出现错误为0. 例如points为空
//切记切记！！！ centers为函数内部申请的内存，用来返回centers_num个点的聚类中心，使用完后一定要释放，记住free(centers)！！！
int  cluster2DPoints( double * points, int points_num, double distance_tolerance, double * & centers, int * centers_num)
{
	double xmax,xmin,ymax,ymin,xdelta,ydelta;
	int nbins_x,nbins_y;
	int x,y;
	int i;
	unsigned int addr,addr2;
	xmax = ymax = 0;
	xmin = ymin = DBL_MAX;
	for( i = 0; i< points_num; i++ )
	{
		addr = 2*i;
		if( points[addr] > xmax)
			xmax = points[addr];
		if( points[addr] < xmin)
			xmin = points[addr];
		if( points[addr+1] > ymax)
			ymax = points[addr+1];
		if( points[addr+1] < ymin)
			ymin = points[addr+1];
	}
	xmax += xmax*0.02;//避免xdelta、ydelta为0
	xmin -= xmin*0.02;
	ymax += ymax*0.02;
	ymin -= ymin*0.02;
	xdelta = (xmax-xmin);
	ydelta = (ymax-ymin);//有问题，假设所有数据一样大，此处为0
	nbins_x = (int)ceil(xdelta/distance_tolerance);
	nbins_y = (int)ceil(ydelta/distance_tolerance);
	if(nbins_x <= 0 )
	{
		nbins_x = 1;//至少保留1个bin
		//error("generateCircleCandidates,nbins_x,nbins_y error");
	}
	if(nbins_y <= 0)
	{
		nbins_y = 1;
	}
	point2d1i * center_bins;
	center_bins = (point2d1i *)calloc(nbins_y*nbins_x, sizeof(point2d1i));//(x,y,z),x用来记sum(xi),y用来记sum(yi),z用来记落在格子里的数量
	memset(center_bins,0,sizeof(point2d1i)*nbins_y*nbins_x);
	if(center_bins == NULL)
		error("cluster2DPoints, not enough memory");
//	cout<<"2D原始数据:"<<points_num<<endl;
	for ( i = 0; i< points_num; i++ )//将圆心记录到格子里面，同时落在相应格子里面的数量++
	{
		addr = 2*i;

//		cout<<points[addr]<<'\t'<<points[addr+1]<<endl;

		x = (int)((points[addr]   - xmin)/xdelta*nbins_x+0.5);//四舍五入
		y = (int)((points[addr+1] - ymin)/ydelta*nbins_y+0.5);
		if( x >= nbins_x)
			x = nbins_x-1;
		if( y >= nbins_y)
			y = nbins_y-1;
		addr2 = y*nbins_x+x;
		center_bins[addr2].x += points[addr];
		center_bins[addr2].y += points[addr+1];
		center_bins[addr2].z ++;
	}
	int initCentersLength = 0;
	for ( y = 0; y<nbins_y; y++)//将vote后非0的格子里面的point取均值，并按照顺序重写到center_bins里面，无内存消耗
		for ( x = 0; x<nbins_x; x++)
		{
			addr = y*nbins_x+x;
			if(center_bins[addr].z > 0)
			{
				center_bins[initCentersLength].x = center_bins[addr].x/center_bins[addr].z;
				center_bins[initCentersLength].y = center_bins[addr].y/center_bins[addr].z;
				initCentersLength++;
			}
		}
	if(initCentersLength == 0)
	{
		(*centers_num) = 0;
		centers = NULL;
		//cout<<"cluster2DPoints,points number:"<<points_num<<endl;
		//cout<<"cluster2DPoints,initCentersLength equals 0"<<endl;
		return 0;
		//error("generateCircleCandidates,initCentersLength equals 0");
	}
	double * initCenters; //initCentersLength x 2
	initCenters = (double*)malloc(sizeof(double)*initCentersLength*2); 
	//将记录在链表里面的分区后的圆心均值记录到数组里，便于作为初始点进行均值漂移
	for ( i = 0; i<initCentersLength; i++ )// initCenters 大小是 initCentersLength*2
	{
		int addr = 2*i;
		initCenters[addr]   = center_bins[i].x;
		initCenters[addr+1] = center_bins[i].y;
	}
	free((void*)center_bins);//赶紧释放该内存

//	cout<<"2D均值漂移前初始迭代点："<<endl;
//	for (int  i = 0; i<initCentersLength; i++)
//		cout<<initCenters[2*i]<<'\t'<<initCenters[2*i+1]<<endl;
	
	//均值漂移的结果会更新到initCenters里面
	meanShift(points,points_num,2,initCenters,initCentersLength,1,distance_tolerance,1e-6,50);//迭代20次

//	cout<<"2D均值漂移后的聚类中心:"<<endl;
//	for (int  i = 0; i<initCentersLength; i++)
//		cout<<initCenters[2*i]<<'\t'<<initCenters[2*i+1]<<endl;

	//聚类
	//千万要注意centers_num是int型指针，++--时要(*centers_num).
	clusterByDistance(initCenters,initCentersLength,2,distance_tolerance/2,40,centers, centers_num);//此处控制参数要改，要调节

//	cout<<"2D距离聚类，去除重复点后的点集:"<<endl;
//	for (int  i = 0; i<(*centers_num); i++)
//		cout<<centers[2*i]<<'\t'<<centers[2*i+1]<<endl;

	if((*centers_num) <= 0)//可无
	{
		return 0;  //不懂为什么，聚类中心的周围确没有最靠近它的点
		//system("pause");
		//error("cluster2DPoints,(*centers_num)<=0");
	}
	free(initCenters);
	//cout<<"2D聚类后数量:"<<(*centers_num)<<endl;
	return 1;
}

//聚类算法，均值漂移
//三个步骤，一是选取初始迭代点，二是均值漂移，三是去除重复点，从而得到聚类中心
//获得候选圆心的聚类中心(xi,yi)
//输入：
//datas，一维点数据,长度为datas_num x 1
//distance_tolerance,数据点聚类的半径
//输出：
//一维数据点的聚类中心 centers是一维double数组， 大小为 centers_num x 1
//正确返回值为1，出现错误为0. 例如points为空
//切记切记！！！ centers为函数内部申请的内存，用来返回centers_num个点的聚类中心，使用完后一定要释放，记住free(centers)！！！
int  cluster1DDatas( double * datas, int datas_num, double distance_tolerance, double * & centers, int * centers_num)
{
	double rmin,rmax,rdelta;
	int r;
	int i;
	rmin = DBL_MAX;
	rmax = 0;
	for( i  = 0; i < datas_num; i++)//将链表里的r集合复制到数组
	{
		if(datas[i] < rmin)//在这一次遍历中，记录最大最小值
			rmin = datas[i];
		if(datas[i] > rmax)
			rmax = datas[i];
	}
	int nbins_r = 0;
	point1d1i * center_bins;
	rmax += rmin*0.02;//避免rmax-rmin = 0
	rmin -= rmin*0.02;
	rdelta = rmax - rmin;
	nbins_r = (int)ceil((rdelta)/distance_tolerance);
	if(nbins_r <= 0)//至少有一个bin
		nbins_r = 1;
	center_bins = (point1d1i *)malloc(sizeof(point1d1i)*nbins_r);
	memset(center_bins,0,sizeof(point1d1i)*nbins_r);//初始化为0
//	cout<<"1D原始数据:"<<datas_num<<endl;
	for( i = 0; i<datas_num; i++)//对分区vote
	{
//		cout<<datas[i]<<endl;
		r = int((datas[i]-rmin)/rdelta*nbins_r+0.5);
		if(r>=nbins_r)
			r = nbins_r-1;
		center_bins[r].data += datas[i];
		center_bins[r].cnt  ++;			
	}
	int init_r_length = 0;
	for( i = 0; i<nbins_r; i++)
	{
		if(center_bins[i].cnt > 0)//统计非0分区,并且对每一个bin取均值，按照顺序重写到center_bins里面，无内存消耗
		{
			center_bins[init_r_length].data = center_bins[i].data/center_bins[i].cnt;
			init_r_length++;
		}
	}
	if(init_r_length == 0)
	{
		(*centers_num) = 0;
		centers = NULL;
		//cout<<"cluster1DDatas,points number:"<<datas_num<<endl;
		//cout<<"cluster2DDatas,init_r_length equals 0"<<endl;
		return 0;
		//error("generateCircleCandidates,initCentersLength equals 0");
	}
	double * initCenters; //init_r_length x 1
	initCenters = (double*)malloc(sizeof(double)*init_r_length); 
	//将记录在链表里面的分区后的圆心均值记录到数组里，便于作为初始点进行均值漂移
	for ( i = 0; i<init_r_length; i++ )// initCenters 大小是 init_r_length x 1
	{
		initCenters[i] = center_bins[i].data;
	}
	free(center_bins);//赶紧释放该内存

//	cout<<"1D均值漂移前初始迭代点："<<endl;
//	for (int  i = 0; i<init_r_length; i++)
//		cout<<initCenters[i]<<'\t';
//	cout<<endl;

	//至此，得到了均值漂移初始的initCenters，为一维double数组，长度是init_r_length
	meanShift(datas, datas_num, 1, initCenters, init_r_length, 1, distance_tolerance, 1e-6, 20);//迭代20次

//	cout<<"1D均值漂移后的聚类中心:"<<endl;
//	for (int  i = 0; i<init_r_length; i++)
//		cout<<initCenters[i]<<'\t';
//	cout<<endl;

	//聚类
	//千万要注意centers_num是int型指针，++--时要(*centers_num).
	clusterByDistance(initCenters, init_r_length, 1, distance_tolerance/2, 40, centers, centers_num);//控制参数40，最多40个点合成1个点
	
//	cout<<"1D距离聚类，去除重复点后的点集:"<<endl;
//	for (int  i = 0; i<(*centers_num); i++)
//		cout<<centers[i]<<'\t';
//	cout<<endl;

	if((*centers_num) <= 0)//可无
	{
		return 0;  //不懂为什么，聚类中心的周围确没有最靠近它的点
		//system("pause");
		//error("cluster1DDatas,(*centers_num)<=0");
	}
    free(initCenters);
//	cout<<"1D聚类后数量::"<<(*centers_num)<<endl;
	return 1;
}

//================================Generate Ellipse Candidates=========================================
//匹配组对，组对的索引参数，椭圆参数
typedef struct PairGroup_s
{
	point2i pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
}PairGroup;

//匹配组对节点
typedef struct PairGroupNode_s
{
	point2i pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
	PairGroupNode_s* next;
}PairGroupNode;

typedef struct  PairGroupList_s
{
	int length;
	PairGroup * pairGroup;
}PairGroupList;

typedef struct Point2dNode_s
{
	point2d point;
	Point2dNode_s * next;
}Point2dNode;

typedef struct Point3dNode_s
{
	point3d point;
	Point3dNode_s * next;
}Point3dNode;

typedef struct Point5dNode_s
{
	point2d center;
	point2d axis;
	double  phi;
	Point5dNode_s * next;
}Point5dNode;

typedef struct Point1dNode_s
{
	double data;
	Point1dNode_s * next;
}Point1dNode;

PairGroupList * pairGroupListInit( int length)
{
	if(length <= 0)
		error("paired groups length less equal than 0");
	PairGroupList * pairGroupList = (PairGroupList*)malloc(sizeof(PairGroupList));
	pairGroupList->length = length;
	pairGroupList->pairGroup = (PairGroup*)malloc(sizeof(PairGroup)*length);
	if(pairGroupList->pairGroup == NULL)
		error("pairGroupListInit,not enough memory");
	return pairGroupList;
}

void freePairGroupList( PairGroupList * list)
{
	if(list == NULL || list->pairGroup == NULL)
		error("freePairGroupList,invalidate free");
	free(list->pairGroup);
	free(list);
	list->pairGroup = NULL;
	list = NULL;
}

//计算梯度，返回模和角度，同时模值太小的像素点直接抑制掉，赋值为NOTDEF
//mod、angles为了传值，是二级指针
void calculateGradient( double * img_in, unsigned int imgx, unsigned int imgy,image_double * mod, image_double * angles)
{
	if(img_in == NULL || imgx == 0 || imgy == 0)
		error("calculateGradient error!");
	(*mod) = new_image_double(imgx,imgy);
	(*angles) = new_image_double(imgx,imgy);
	double threshold = 2/sin(22.5/180*M_PI);
	unsigned int x,y,adr;
	double com1,com2;
	double gx,gy;
	double norm,norm_square;
	double sum = 0;

	//double max_grad = 0.0;
	//边界初始为NOTDEF
	for ( x = 0; x<imgx; x++) 
	{
		//(*angles)->data[x]=NOTDEF;
		(*angles)->data[(imgy-1)*imgx+x]=NOTDEF;
		//(*mod)->data[x]=NOTDEF;
		(*mod)->data[(imgy-1)*imgx+x]=NOTDEF;
	}
	for ( y = 0; y<imgy; y++) 
	{
		//(*angles)->data[y*imgx] = NOTDEF;
		(*angles)->data[y*imgx+imgx-1] = NOTDEF;
		//(*mod)->data[y*imgx] = NOTDEF;
		(*mod)->data[y*imgx+imgx-1] = NOTDEF;
	}
	 /* compute gradient on the remaining pixels */
	for(x=0;x<imgx-1;x++)
		for(y=0;y<imgy-1;y++)
		{
			adr = y*imgx+x;
		  /*
		     Norm 2 computation using 2x2 pixel window:
		       A B
		       C D
		     and
		       com1 = D-A,  com2 = B-C.
		     Then
		       gx = B+D - (A+C)   horizontal difference
		       gy = C+D - (A+B)   vertical difference
		     com1 and com2 are just to avoid 2 additions.
		   */
		  com1 = img_in[adr+imgx+1] - img_in[adr];
		  com2 = img_in[adr+1]   - img_in[adr+imgx];

		  gx = com1+com2; /* gradient x component */
		  gy = com1-com2; /* gradient y component */
		  norm_square = gx*gx+gy*gy;

		  norm = sqrt( norm_square / 4.0 ); /* gradient norm */

		  (*mod)->data[adr] = norm; /* store gradient norm */

		  if( norm <= threshold ) /* norm too small, gradient no defined */
		  {
		    (*angles)->data[adr] = NOTDEF; /* gradient angle not defined */
			(*mod)->data[adr] = NOTDEF;
		  }
		  else
		    {
		      /* gradient angle computation */
		      (*angles)->data[adr] = atan2(gx,-gy);
		    }
		}
}

void calculateGradient2( double * img_in, unsigned int imgx, unsigned int imgy, image_double * angles)
{
	if(img_in == NULL || imgx == 0 || imgy == 0)
		error("calculateGradient error!");
	image_double mod = new_image_double(imgx,imgy);
	(*angles) = new_image_double(imgx,imgy);
	unsigned int x,y,adr;
	double com1,com2;
	double gx,gy;
	double norm,norm_square;
	double threshold;
	double sum = 0;
	double value;  
	//double max_grad = 0.0;
	//边界初始为NOTDEF
	for ( x = 0; x<imgx; x++) 
	{
		(*angles)->data[x]=NOTDEF;
		(*angles)->data[(imgy-1)*imgx+x]=NOTDEF;
		(mod)->data[x]=NOTDEF;
		(mod)->data[(imgy-1)*imgx+x]=NOTDEF;
	}
	for ( y = 0; y<imgy; y++) 
	{
		(*angles)->data[y*imgx] = NOTDEF;
		(*angles)->data[y*imgx+imgx-1] = NOTDEF;
		(mod)->data[y*imgx] = NOTDEF;
		(mod)->data[y*imgx+imgx-1] = NOTDEF;
	}
	 /* compute gradient on the remaining pixels */
	for(x=1;x<imgx-1;x++)
		for(y=1;y<imgy-1;y++)
		{
			adr = y*imgx+x;
		  /*
		     Norm 2 computation using 2x2 pixel window:
		       A B C
		       D E F
			   G H I
		     and
		       com1 = C-G,  com2 = I-A.
		     Then
		       gx = C+2F+I - (A+2D+G)=com1+com2+2(F-D)   horizontal difference
		       gy = G+2H+I - (A+2B+C)=-com1+com2+2(H-B)   vertical difference
		     com1 and com2 are just to avoid 2 additions.
		   */
		  com1 = img_in[adr-imgx+1] - img_in[adr+imgx-1];
		  com2 = img_in[adr+imgx+1] - img_in[adr-imgx-1];

		  gx = (com1+com2+2*(img_in[adr+1] - img_in[adr-1]))/(8.0*255); /* gradient x component */
		  gy = (-com1+com2+2*(img_in[adr+imgx] - img_in[adr-imgx]))/(8.0*255); /* gradient y component */
		  norm_square = gx*gx+gy*gy;
		  sum+=norm_square;

		  norm = sqrt( norm_square); /* gradient norm */

		  (mod)->data[adr] = norm; /* store gradient norm */
		   /* gradient angle computation */
	     (*angles)->data[adr] = atan2(gy,gx);
		}
	threshold = 2*sqrt(sum/(imgx*imgy));//自动阈值
	//non maximum suppression
	for(x=1;x<imgx-1;x++)
		for(y=1;y<imgy-1;y++)
		{
			adr = y*imgx+x;
			value = (*angles)->data[adr];
			if((mod)->data[adr] < threshold )
			{
				(*angles)->data[adr] = NOTDEF;
				continue;
			}
			if( (value > -M_1_8_PI && value<=M_1_8_PI) || (value <= -M_7_8_PI ) || (value > M_7_8_PI))
			{
				if((mod)->data[adr] <= (mod)->data[adr+1] || (mod)->data[adr] <= (mod)->data[adr-1])
					(*angles)->data[adr] = NOTDEF;
			}
			else if( (value> M_1_8_PI && value<= M_3_8_PI) || (value> -M_7_8_PI && value<= -M_5_8_PI) )
			{
				if((mod)->data[adr] <= (mod)->data[adr-imgx-1] || (mod)->data[adr] <= (mod)->data[adr+imgx+1])
					(*angles)->data[adr] = NOTDEF;
			}
			else if((value> M_3_8_PI && value<= M_5_8_PI) || (value> -M_5_8_PI && value<= -M_3_8_PI))
			{
				if((mod)->data[adr] <= (mod)->data[adr-imgx] || (mod)->data[adr] <= (mod)->data[adr+imgx])
					(*angles)->data[adr] = NOTDEF;
			}
			else 
			{
				if((mod)->data[adr] <= (mod)->data[adr-imgx+1] || (mod)->data[adr] <= (mod)->data[adr+imgx-1])
					(*angles)->data[adr] = NOTDEF;
			}
		}
    //也标记到mod图上面
	//for(x=1;x<imgx-1;x++)
	//	for(y=1;y<imgy-1;y++)
	//	{
	//		if((*angles)->data[y*imgx+x] == NOTDEF)
	//			(mod)->data[y*imgx+x] = NOTDEF;
	//	}
		free_image_double(mod);
}

//=============================================================================
//需要包含如下头文件
//#include <opencv2\opencv.hpp>
//using namespace cv;
void cvCanny3(	const void* srcarr, void* dstarr,
				void* dxarr, void* dyarr,
                int aperture_size )
{
    //cv::Ptr<CvMat> dx, dy;
    cv::AutoBuffer<char> buffer;
    std::vector<uchar*> stack;
    uchar **stack_top = 0, **stack_bottom = 0;

    CvMat srcstub, *src = cvGetMat( srcarr, &srcstub );
    CvMat dststub, *dst = cvGetMat( dstarr, &dststub );

	CvMat dxstub, *dx = cvGetMat( dxarr, &dxstub );
	CvMat dystub, *dy = cvGetMat( dyarr, &dystub );


    CvSize size;
    int flags = aperture_size;
    int low, high;
    int* mag_buf[3];
    uchar* map;
    ptrdiff_t mapstep;
    int maxsize;
    int i, j;
    CvMat mag_row;

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1 ||
		CV_MAT_TYPE( dx->type  ) != CV_16SC1 ||
		CV_MAT_TYPE( dy->type  ) != CV_16SC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "" );
	
    aperture_size &= INT_MAX;
    if( (aperture_size & 1) == 0 || aperture_size < 3 || aperture_size > 7 )
        CV_Error( CV_StsBadFlag, "" );


	size.width = src->cols;
    size.height = src->rows;

	//aperture_size = -1; //SCHARR
    cvSobel( src, dx, 1, 0, aperture_size );
    cvSobel( src, dy, 0, 1, aperture_size );

	Mat1f magGrad(size.height, size.width, 0.f);
	float maxGrad(0);
	float val(0);
	for(i=0; i<size.height; ++i)
	{
		float* _pmag = magGrad.ptr<float>(i);
		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
		for(j=0; j<size.width; ++j)
		{
			val = float(abs(_dx[j]) + abs(_dy[j]));
			_pmag[j] = val;
			maxGrad = (val > maxGrad) ? val : maxGrad;
		}
	}
	
	//% Normalize for threshold selection
	//normalize(magGrad, magGrad, 0.0, 1.0, NORM_MINMAX);

	//% Determine Hysteresis Thresholds
	
	//set magic numbers
	const int NUM_BINS = 64;	
	const double percent_of_pixels_not_edges = 0.9;
	const double threshold_ratio = 0.3;

	//compute histogram
	int bin_size = cvFloor(maxGrad / float(NUM_BINS) + 0.5f) + 1;
	if (bin_size < 1) bin_size = 1;
	int bins[NUM_BINS] = { 0 }; 
	for (i=0; i<size.height; ++i) 
	{
		float *_pmag = magGrad.ptr<float>(i);
		for(j=0; j<size.width; ++j)
		{
			int hgf = int(_pmag[j]);
			bins[int(_pmag[j]) / bin_size]++;
		}
	}	

	
	

	//% Select the thresholds
	float total(0.f);	
	float target = float(size.height * size.width * percent_of_pixels_not_edges);
	int low_thresh, high_thresh(0);
	
	while(total < target)
	{
		total+= bins[high_thresh];
		high_thresh++;
	}
	high_thresh *= bin_size;
	low_thresh = cvFloor(threshold_ratio * float(high_thresh));
	
    if( flags & CV_CANNY_L2_GRADIENT )
    {
        Cv32suf ul, uh;
        ul.f = (float)low_thresh;
        uh.f = (float)high_thresh;

        low = ul.i;
        high = uh.i;
    }
    else
    {
        low = cvFloor( low_thresh );
        high = cvFloor( high_thresh );
    }

    
	buffer.allocate( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int) );
    mag_buf[0] = (int*)(char*)buffer;
    mag_buf[1] = mag_buf[0] + size.width + 2;
    mag_buf[2] = mag_buf[1] + size.width + 2;
    map = (uchar*)(mag_buf[2] + size.width + 2);
    mapstep = size.width + 2;

    maxsize = MAX( 1 << 10, size.width*size.height/10 );
    stack.resize( maxsize );
    stack_top = stack_bottom = &stack[0];

    memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
    memset( map, 1, mapstep );
    memset( map + mapstep*(size.height + 1), 1, mapstep );

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    mag_row = cvMat( 1, size.width, CV_32F );

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int* _mag = mag_buf[(i > 0) + 1] + 1;
        float* _magf = (float*)_mag;
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        uchar* _map;
        int x, y;
        ptrdiff_t magstep1, magstep2;
        int prev_flag = 0;

        if( i < size.height )
        {
            _mag[-1] = _mag[size.width] = 0;

            if( !(flags & CV_CANNY_L2_GRADIENT) )
                for( j = 0; j < size.width; j++ )
                    _mag[j] = abs(_dx[j]) + abs(_dy[j]);

            else
            {
                for( j = 0; j < size.width; j++ )
                {
                    x = _dx[j]; y = _dy[j];
                    _magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
                }
            }
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(int) );

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;

        _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;

        _mag = mag_buf[1] + 1; // take the central row
        _dx = (short*)(dx->data.ptr + dx->step*(i-1));
        _dy = (short*)(dy->data.ptr + dy->step*(i-1));

        magstep1 = mag_buf[2] - mag_buf[1];
        magstep2 = mag_buf[0] - mag_buf[1];

        if( (stack_top - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        for( j = 0; j < size.width; j++ )
        {
            #define CANNY_SHIFT 15
            #define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

            x = _dx[j];
            y = _dy[j];
            int s = x ^ y;
            int m = _mag[j];

            x = abs(x);
            y = abs(y);
            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = (uchar)1;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( (stack_top - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep-1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep+1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep-1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep+1] )
            CANNY_PUSH( m + mapstep + 1 );
    }

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep*(i+1) + 1;
        uchar* _dst = dst->data.ptr + dst->step*i;

        for( j = 0; j < size.width; j++ )
		{
            _dst[j] = (uchar)-(_map[j] >> 1);
		}
	}
};

void Canny3(	InputArray image, OutputArray _edges,
				OutputArray _sobel_x, OutputArray _sobel_y,
                int apertureSize, bool L2gradient )
{
    Mat src = image.getMat();
    _edges.create(src.size(), CV_8U);
	_sobel_x.create(src.size(), CV_16S);
	_sobel_y.create(src.size(), CV_16S);


    CvMat c_src = src, c_dst = _edges.getMat();
	CvMat c_dx = _sobel_x.getMat();
	CvMat c_dy = _sobel_y.getMat();


    cvCanny3(	&c_src, &c_dst, 
				&c_dx, &c_dy,
				apertureSize + (L2gradient ? CV_CANNY_L2_GRADIENT : 0));
};

//canny
void calculateGradient3( double * img_in, unsigned int imgx, unsigned int imgy, image_double * angles)
{
	Mat1b edge;
	Mat1s DX,DY;
	Mat1b gray = Mat::zeros(imgy,imgx,CV_8UC1);
	unsigned int x,y,addr;
	(*angles) = new_image_double(imgx,imgy);
	//copy to gray image
	for ( y = 0; y<imgy; y++)
		for ( x = 0; x<imgx; x++)
		{
			addr = y*imgx+x;
			gray.data[addr] = (uchar)(img_in[addr]);
		}
	//canny
   Canny3(gray,edge,DX,DY,3,false);
   for ( y = 0; y<imgy; y++)
   {
	    short* _dx = DX.ptr<short>(y);
		short* _dy = DY.ptr<short>(y);
		uchar* _e = edge.ptr<uchar>(y);
		for ( x = 0; x<imgx; x++)
		{
			if(_e[x] > 0)//0 or 255
			{
				(*angles)->data[y*imgx+x]  = atan2((double)_dy[x],(double)_dx[x]);//calculate gradient 
			}
			else
				(*angles)->data[y*imgx+x] = NOTDEF;
		}
   }
   edge.release();
   DX.release();
   DY.release();
   gray.release();
}


//=============================================================================
/** Convert ellipse from matrix form to common form:
    ellipse = (centrex,centrey,ax,ay,orientation).
 */
int ellipse2Param(double *p,double param[])
{
	// ax^2 + bxy + cy^2 + dx + ey + f = 0 
  double a,b,c,d,e,f;
  double thetarad,cost,sint,cos_squared,sin_squared,cos_sin,Ao,Au,Av,Auu,Avv,tuCentre,tvCentre,wCentre,uCentre,vCentre,Ru,Rv;
  a = p[0];
  b = p[1];
  c = p[2];
  d = p[3];
  e = p[4];
  f = p[5]; 

  thetarad=0.5*atan2(b,a-c); 
  cost=cos(thetarad);
  sint=sin(thetarad);
  sin_squared=sint*sint;
  cos_squared=cost*cost;
  cos_sin=sint*cost;
  Ao=f;
  Au=d*cost+e* sint;
  Av=-d*sint+e* cost;
  Auu=a*cos_squared+c*sin_squared+b*cos_sin;
  Avv=a*sin_squared+c*cos_squared-b*cos_sin;

  if(Auu==0 || Avv==0){ param[0]=0;param[1]=0;param[2]=0;param[3]=0;param[4]=0;return 0;}
  else
    {
      tuCentre=-Au/(2.*Auu);
      tvCentre=-Av/(2.*Avv);
      wCentre=Ao-Auu*tuCentre*tuCentre-Avv*tvCentre*tvCentre;
      uCentre=tuCentre*cost-tvCentre*sint;
      vCentre=tuCentre*sint+tvCentre*cost;
      Ru=-wCentre/Auu;
      Rv=-wCentre/Avv;
 //     if (Ru>0) Ru=pow(Ru,0.5);
 //     else Ru=-pow(-Ru,0.5);
 //     if (Rv>0) Rv=pow(Rv,0.5);
 //     else Rv=-pow(-Rv,0.5);
	  if (Ru <= 0 || Rv <= 0)//长短轴小于0的情况？？？
		  return 0;
	  Ru = sqrt(Ru);
	  Rv = sqrt(Rv);
      param[0]=uCentre;param[1]=vCentre;
      param[2]=Ru;param[3]=Rv;param[4]=thetarad;
	  //会出现Ru < Rv情况，对调一下
	  if(Ru < Rv )
	  {
		  param[2] = Rv;
		  param[3] = Ru;
		  if(thetarad < 0)//调换长短轴，使得第三个参数为长轴，第四个为短轴
			  param[4] += M_1_2_PI;
		  else
			  param[4] -= M_1_2_PI;
		  if(thetarad < - M_1_2_PI)//长轴倾角限定在-pi/2 ~ pi/2，具备唯一性
			  param[4] += M_PI;
		  if(thetarad > M_1_2_PI)
			  param[4] -= M_PI;
	  }
    }
  return 1;
}
//input : (xi,yi)
//output: x0,y0,a,b,phi,ellipara需要事先申请内存
//successfull, return 1; else return 0
int fitEllipse(point2d* dataxy, int datanum, double* ellipara)
{
	double* D = (double*)malloc(datanum*6*sizeof(double));
	double S[36]; 
	double C[36];
	memset(D,0,sizeof(double)*datanum);
	memset(S,0,sizeof(double)*36);
	memset(C,0,sizeof(double)*36);
	for ( int i = 0; i<datanum; i++)
	{
		D[i*6]  = dataxy[i].x*dataxy[i].x;
		D[i*6+1]= dataxy[i].x*dataxy[i].y;
		D[i*6+2]= dataxy[i].y*dataxy[i].y;
		D[i*6+3]= dataxy[i].x;
		D[i*6+4]= dataxy[i].y;
		D[i*6+5]= 1;
	}
	for ( int i = 0; i<6; i++)
		for ( int j = i; j<6; j++)
		{
			//S[i*6+j]
			for ( int k = 0; k<datanum; k++)
				S[i*6+j] += D[k*6+i]*D[k*6+j];
		}
	free(D);//释放内存
	//对称矩阵赋值
	for ( int i = 0; i<6; i++)
		for ( int j = 0; j<i; j++)
			S[i*6+j]=S[j*6+i];
	C[0*6+2] = 2;
	C[1*6+1] = -1;
	C[2*6+0] = 2;
	// eig(S,C) eig(inv(S)*C)
	double alphar[6],alphai[6],beta[6];
	double vl[36] = {0};//此处不用
	double vr[36] = {0};
	char JOBVL = 'N';
	char JOBVR = 'V';
	ptrdiff_t fitN = 6;
	double fitWork[64];
	ptrdiff_t workLen = 64;
	ptrdiff_t info;
	//info = LAPACKE_dggev(LAPACK_ROW_MAJOR,'N','V',6,S,6,C,6,alphar,alphai,beta,vl,6,vr,6);
	//注意S为对称矩阵，故转置后等于本身，变成列优先，S可以不变
	dggev(&JOBVL,&JOBVR,&fitN,S,&fitN,C,&fitN,alphar,alphai,beta,vl,&fitN,vr,&fitN,fitWork,&workLen,&info);
	if(info == 0)
	{
		int index = -1;
		for ( int i = 0; i<6; i++)
			if( (alphar[i]>=-(2.2204460492503131e-014)) && (alphai[i] == 0) && (beta[i] != 0)) // 100*DBL_EPSILON, eigenvalue = (alphar + i*alphai)/beta
				index = i;//vr[:,i],vr第i列对应的特征向量则为拟合参数
		if(index == -1)//再试一次，放宽对实部>0的约束，放宽到>-0.005
		{
			double temp = -0.005;//这个参数很关键
			for ( int i = 0; i<6; i++)
			if( (alphar[i]>=temp) && (alphai[i] == 0) && (beta[i] != 0)) // 100*DBL_EPSILON, eigenvalue = (alphar + i*alphai)/beta
			{
				temp = alphar[i];
				index = i;//vr[:,i],vr第i列对应的特征向量则为拟合参数
			}
		}
		if(index != -1)
		{
			//此处借用beta来传递参数
		    //beta[0] = vr[6*0+index];
		    //beta[1] = vr[6*1+index];
		    //beta[2] = vr[6*2+index];
		    //beta[3] = vr[6*3+index];
		    //beta[4] = vr[6*4+index];
		    //beta[5] = vr[6*5+index];
			  beta[0] = vr[6*index+0];
		      beta[1] = vr[6*index+1];
		      beta[2] = vr[6*index+2];
		      beta[3] = vr[6*index+3];
		      beta[4] = vr[6*index+4];
		      beta[5] = vr[6*index+5];
			ellipse2Param(beta,ellipara);//ax^2 + bxy + cy^2 + dx + ey + f = 0, transform to (x0,y0,a,b,phi)
			return 1;
		}
	}
	return 0;
}

//input: dataxy为数据点(xi,yi),总共有datanum个
//output: 拟合矩阵S. 注意：S需要事先申请内存，double S[36].
inline void calcuFitMatrix(point2d* dataxy, int datanum, double * S)
{
	double* D = (double*)malloc(datanum*6*sizeof(double));
	memset(D,0,sizeof(double)*datanum);
	for ( int i = 0; i<datanum; i++)
	{
		D[i*6]  = dataxy[i].x*dataxy[i].x;
		D[i*6+1]= dataxy[i].x*dataxy[i].y;
		D[i*6+2]= dataxy[i].y*dataxy[i].y;
		D[i*6+3]= dataxy[i].x;
		D[i*6+4]= dataxy[i].y;
		D[i*6+5]= 1;
	}
	for ( int i = 0; i<6; i++)
	{
		for ( int j = i; j<6; j++)
		{
			//S[i*6+j]
			for ( int k = 0; k<datanum; k++)
				S[i*6+j] += D[k*6+i]*D[k*6+j];
		}
	}
    free(D);//释放内存
	//对称矩阵赋值
	for ( int i = 0; i<6; i++)
		for ( int j = 0; j<i; j++)
			S[i*6+j]=S[j*6+i];
}
//input: fit matrixes S1,S2. length is 36.
//output: fit matrix S_out. S_out = S1 + S2.
//S_out事先需要申请内存
inline void addFitMatrix(double * S1, double * S2, double * S_out)
{
	int ind;
	for ( int i = 0; i<6; i++ )
		for ( int j = i; j<6; j++)
		{
			ind = i*6+j;
			S_out[ind] = S1[ind]+S2[ind];
		}
	//对称矩阵赋值
	for ( int i = 0; i<6; i++)
		for ( int j = 0; j<i; j++)
			S_out[i*6+j]=S_out[j*6+i];
}
//input : S矩阵，6 x 6 = 36
//output: (A,B,C,D,E,F)且A>0, ellicoeff需要事先申请内存. 当要转换成(x0,y0,a,b,phi)时，则要用
//ellipse2Param(ellicoeff,ellipara); ax^2 + bxy + cy^2 + dx + ey + f = 0, transform to (x0,y0,a,b,phi)
//successfull, return 1; else return 0
int fitEllipse2(double * S, double* ellicoeff)
{
	double C[36];
	memset(C,0,sizeof(double)*36);
	
	C[0*6+2] = 2;
	C[1*6+1] = -1;
	C[2*6+0] = 2;
	// eig(S,C) eig(inv(S)*C)
	double alphar[6],alphai[6],beta[6];
	double vl[36] = {0};//此处不用
	double vr[36] = {0};
	char JOBVL = 'N';
	char JOBVR = 'V';
	ptrdiff_t fitN = 6;
	double fitWork[64];
	ptrdiff_t workLen = 64;
	ptrdiff_t info;
	//info = LAPACKE_dggev(LAPACK_ROW_MAJOR,'N','V',6,S,6,C,6,alphar,alphai,beta,vl,6,vr,6);
	dggev(&JOBVL,&JOBVR,&fitN,S,&fitN,C,&fitN,alphar,alphai,beta,vl,&fitN,vr,&fitN,fitWork,&workLen,&info);
	if(info == 0)
	{
		int index = -1;
		for ( int i = 0; i<6; i++)
			if( (alphar[i]>=-(2.2204460492503131e-014)) && (alphai[i] == 0) && (beta[i] != 0)) // 100*DBL_EPSILON, eigenvalue = (alphar + i*alphai)/beta
				index = i;//vr[:,i],vr第i列对应的特征向量则为拟合参数
		if(index == -1)//再试一次，放宽对实部>0的约束，放宽到>-0.005
		{
			double temp = -0.005;//这个参数很关键
			for ( int i = 0; i<6; i++)
			if( (alphar[i]>=temp) && (alphai[i] == 0) && (beta[i] != 0)) // 100*DBL_EPSILON, eigenvalue = (alphar + i*alphai)/beta
			{
				temp = alphar[i];
				index = i;//vr[:,i],vr第i列对应的特征向量则为拟合参数
			}
		}
		if(index != -1)
		{
			//此处借用beta来传递参数
	        if(vr[6*index+0] < 0)//注意列优先
			{
				ellicoeff[0] = -vr[6*index+0]; //-vr[6*0+index];
				ellicoeff[1] = -vr[6*index+1]; //-vr[6*1+index];
				ellicoeff[2] = -vr[6*index+2]; //-vr[6*2+index];
				ellicoeff[3] = -vr[6*index+3]; //-vr[6*3+index];
				ellicoeff[4] = -vr[6*index+4]; //-vr[6*4+index];
				ellicoeff[5] = -vr[6*index+5]; //-vr[6*5+index];
			}
			else
			{
				ellicoeff[0] = vr[6*index+0];//vr[6*0+index];
				ellicoeff[1] = vr[6*index+1];//vr[6*1+index];
				ellicoeff[2] = vr[6*index+2];//vr[6*2+index];
				ellicoeff[3] = vr[6*index+3];//vr[6*3+index];
				ellicoeff[4] = vr[6*index+4];//vr[6*4+index];
				ellicoeff[5] = vr[6*index+5];//vr[6*5+index];
			}
			return 1;
		}
	}
	return 0;
}

//入参：e1 = (x1,y1,a1,b1,phi1), e2 = (x2,y2,a2,b2,phi2)
//输出：相等为1，否则为0
inline bool isEllipseEqual(double * ellipse1, double * ellipse2, double centers_distance_threshold, double semimajor_errorratio, double semiminor_errorratio, double angle_errorratio, double iscircle_ratio)
{
	bool con1 = ( abs(ellipse1[0] - ellipse2[0]) < centers_distance_threshold && abs(ellipse1[1] - ellipse2[1]) < centers_distance_threshold &&
		abs(ellipse1[2] - ellipse2[2])/MAX(ellipse1[2],ellipse2[2]) < semimajor_errorratio && abs(ellipse1[3] - ellipse2[3])/MIN(ellipse1[3],ellipse2[3]) < semiminor_errorratio );
	bool con2 = ( ellipse1[3]/ellipse1[2] >= iscircle_ratio );//0.9 0.85
	bool con3 = ( ellipse2[3]/ellipse2[2] >= iscircle_ratio );
	bool con4 = ( (con2 && con3) || (con2 == false && con3 == false && abs(ellipse1[4]-ellipse2[4])<= angle_errorratio*M_PI) );
	return (con1 && con4);
}

inline bool regionLimitation( point2d point_g1s, point2d g1s_ls_dir, point2d point_g1e, point2d g1e_ls_dir, point2d point_g2s, point2d g2s_ls_dir, point2d point_g2e, point2d g2e_ls_dir, double polarity, double region_limitation_dis_tolerance)
{
	point2d g1m_ls_dir, g2m_ls_dir;
	point2d g1s_arc_dir,g1e_arc_dir,g1m_arc_dir,g2s_arc_dir,g2e_arc_dir,g2m_arc_dir;
	point2d test_vec1,test_vec2,test_vec3; //弧指向圆心的向量和测试向量
	//组的pend<-pstart构成的向量为gim_arc_dir
	double xdelta, ydelta, theta;
	xdelta = point_g1e.x - point_g1s.x;
	ydelta = point_g1e.y - point_g1s.y;
	theta = atan2(ydelta,xdelta);
	g1m_ls_dir.x = cos(theta);
	g1m_ls_dir.y = sin(theta);
	xdelta = point_g2e.x - point_g2s.x;
	ydelta = point_g2e.y - point_g2s.y;
	theta = atan2(ydelta,xdelta);
	g2m_ls_dir.x = cos(theta);
	g2m_ls_dir.y = sin(theta);

	if( polarity == 1)// polarity is equal 1, arc vector = (dy,-dx)
	{
		g1s_arc_dir.x = g1s_ls_dir.y;
		g1s_arc_dir.y = -g1s_ls_dir.x;
		g1e_arc_dir.x = g1e_ls_dir.y;
		g1e_arc_dir.y = -g1e_ls_dir.x;
		g1m_arc_dir.x = g1m_ls_dir.y;
		g1m_arc_dir.y = -g1m_ls_dir.x;
		g2s_arc_dir.x = g2s_ls_dir.y;
		g2s_arc_dir.y = -g2s_ls_dir.x;
		g2e_arc_dir.x = g2e_ls_dir.y;
		g2e_arc_dir.y = -g2e_ls_dir.x;
		g2m_arc_dir.x = g2m_ls_dir.y;
		g2m_arc_dir.y = -g2m_ls_dir.x;
	}
	else// polarity is equal -1, arc vector = (-dy,dx)
	{
		g1s_arc_dir.x = -g1s_ls_dir.y;
		g1s_arc_dir.y = g1s_ls_dir.x;
		g1e_arc_dir.x = -g1e_ls_dir.y;
		g1e_arc_dir.y = g1e_ls_dir.x;
		g1m_arc_dir.x = -g1m_ls_dir.y;
		g1m_arc_dir.y = g1m_ls_dir.x;
		g2s_arc_dir.x = -g2s_ls_dir.y;
		g2s_arc_dir.y = g2s_ls_dir.x;
		g2e_arc_dir.x = -g2e_ls_dir.y;
		g2e_arc_dir.y = g2e_ls_dir.x;
		g2m_arc_dir.x = -g2m_ls_dir.y;
		g2m_arc_dir.y = g2m_ls_dir.x;
	}
	test_vec1.x = (point_g2e.x - point_g1s.x);
	test_vec1.y = (point_g2e.y - point_g1s.y);
	test_vec2.x = (point_g2s.x - point_g1e.x);
	test_vec2.y = (point_g2s.y - point_g1e.y);
	test_vec3.x = (test_vec1.x + test_vec2.x)/2;
	test_vec3.y = (test_vec1.y + test_vec2.y)/2;
	double t1,t2,t3,t4,t5,t6;
	t1 = dotProduct(g1s_arc_dir,test_vec1);
	t2 = dotProduct(g1e_arc_dir,test_vec2);
	t3 = dotProduct(g1m_arc_dir,test_vec3);
	t4 = -dotProduct(g2e_arc_dir,test_vec1);
	t5 = -dotProduct(g2s_arc_dir,test_vec2);
	t6 = -dotProduct(g2m_arc_dir,test_vec3);

	if(  dotProduct(g1s_arc_dir,test_vec1)  >= region_limitation_dis_tolerance && \
		 dotProduct(g1e_arc_dir,test_vec2)  >= region_limitation_dis_tolerance && \
		 dotProduct(g1m_arc_dir,test_vec3)  >= region_limitation_dis_tolerance && \
		-dotProduct(g2e_arc_dir,test_vec1) >= region_limitation_dis_tolerance && \
	    -dotProduct(g2s_arc_dir,test_vec2) >= region_limitation_dis_tolerance && \
		-dotProduct(g2m_arc_dir,test_vec3) >= region_limitation_dis_tolerance
		)
		return TRUE;
	return FALSE;
}

/*
void drawEllipse(Mat img, double * ellipara)
{
  Point peliicenter(ellipara[0],ellipara[1]);
  Size  saxis(ellipara[2],ellipara[3]);
  //Mat ellimat = Mat::zeros(img.rows,img.cols,CV_8UC3);
  //ellimat.setTo(255);
  static int ccc = 0;
  static unsigned int cnt = 0;
  if(cnt % 2 == 0 )
	  ccc = 0;
  else
  {
	  ccc = 255;
	  cout<<cnt/2<<'\t'<<ellipara[0]<<'\t'<<ellipara[1]<<"\t"<<ellipara[2]<<'\t'<<ellipara[3]<<'\t'<<ellipara[4]<<endl;
  }
  cnt++;

  Mat imgtemp = img.clone();
  ellipse(imgtemp,peliicenter,saxis,ellipara[4]*180/M_PI,0,360,(Scalar(0,255,ccc)),2);
  namedWindow("w1");
  imshow("w1",imgtemp);
  //waitKey(0);
}
void drawEdge(Mat img, point2d * dataxy, int num)
{
	 static int ccc = 0;
     static int cnt = 0;
     cnt++;
     if(cnt % 2 == 0 )
	     ccc = 0;
     else
	  ccc = 255;
	Mat imgtemp = img.clone();
	for (int i = 0; i<num; i++)
	{
		imgtemp.at<Vec3b>(dataxy[i].y,dataxy[i].x) = (Vec3b(ccc,255,0));
	}
	namedWindow("w2");
    imshow("w2",imgtemp);
}
*/

/*----------------------------------------------------------------------------*/
/** Approximate the distance between a point and an ellipse using Rosin distance.
 */
inline double d_rosin (double *param, double x, double y)
{ 
  double ae2 = param[2]*param[2];
  double be2 = param[3]*param[3];
  x = x - param[0];
  y = y - param[1];
  double xp = x*cos(-param[4])-y*sin(-param[4]);
  double yp = x*sin(-param[4])+y*cos(-param[4]);
  double fe2;
  fe2 = ae2-be2;
  double X = xp*xp;
  double Y = yp*yp;
  double delta = (X+Y+fe2)*(X+Y+fe2)-4*X*fe2;
  double A = (X + Y + fe2 - sqrt(delta))/2.0; 
  double ah = sqrt(A);
  double bh2 = fe2-A;
  double term = (A*be2+ae2*bh2);
  double xi = ah*sqrt(ae2*(be2+bh2)/term);
  double yi = param[3]*sqrt(bh2*(ae2-A)/term);
  double d[4],dmin;


  d[0] = dist(xp,yp,xi,yi);
  d[1] = dist(xp,yp,xi,-yi);
  d[2] = dist(xp,yp,-xi,yi);
  d[3] = dist(xp,yp,-xi,-yi);
  dmin = DBL_MAX;
  for ( int i = 0; i<4; i++)
  {
	  if( d[i] <= dmin)
		  dmin = d[i];
  }
//  if (X+Y>xi*xi+yi*yi)
//    return dmin;
//  else return -dmin; 
  return dmin;
}
/*----------------------------------------------------------------------------*/

//输入
//lsd算法检测得到的线段集合lines的数量line_num，return的返回值是line_nums条线段，为一维double型数组lines，长度为8*n，每8个为一组
//存着x1,y1,x2,y2,dx,dy,length,polarity
//groups: 线段分组，每个组存按照几何分布顺序顺时针或者逆时针存储着线段索引，线段索引范围是0~line_num-1. 这里由于是指针，使用时要注意(*group)
//first_group_ind、second_group_ind是匹配组队的索引，当提取salient hypothesis时，second_group_ind = -1, fit_matrix2 = NULL.
//fit_matrix1, fit_matrix2, 分别是组队的对应的拟合矩阵
//angles, 是边缘点图+梯度方向。 无边缘点时是NODEF
//distance_tolerance:
//group_inliers_num:记录着各个组的支持内点数量的数组，实时更新，初始时为0
//输出
//ellipara
bool calcEllipseParametersAndValidate( double * lines, int line_num, vector<vector<int>> * groups, int first_group_ind,int second_group_ind, double * fit_matrix1, double * fit_matrix2, image_double angles, double distance_tolerance, unsigned int * group_inliers_num, point5d *ellipara)
{
	double S[36]; //拟合矩阵S
	double Coefficients[6] = {0,0,0,0,0,0};// ax^2 + bxy + cy^2 + dx + ey + f = 0 
	double param[5], param2[5];
	int info,addr;
	rect rec;
	rect_iter* iter;
	int rec_support_cnt,rec_inliers_cnt;
	bool flag1 = TRUE, flag2 = TRUE;
	double point_normalx, point_normaly, point_normal, temp;
	vector<point2i> first_group_inliers, second_group_inliers;
	point2i pixel_temp;
	double semimajor_errorratio,semiminor_errorratio,iscircle_ratio;
	if( fit_matrix2 == NULL || second_group_ind == -1)//只对一个覆盖度较大的组进行拟合
	{
		for ( int i  = 0; i < 36; i++)
			S[i] = fit_matrix1[i];
	}
	else
	{
		addFitMatrix(fit_matrix1,fit_matrix2,S);//对组对进行拟合， S = fit_matrix1 + fit_matrix2
	}
	info = fitEllipse2(S, Coefficients);// ax^2 + bxy + cy^2 + dx + ey + f = 0, a > 0
	if ( info == 0 )//拟合失败
	{
		ellipara = NULL;
		return FALSE;
	}
	ellipse2Param(Coefficients, param);// (x0,y0,a,b,phi)
	if ( min(param[2],param[3]) < 3*distance_tolerance || max(param[2],param[3]) > min(angles->xsize,angles->ysize) ||  param[0] < 0 || param[0] > angles->xsize || param[1] < 0 || param[1] > angles->ysize )
	{
		ellipara = NULL;
		return FALSE;
	}
	//if ( first_group_ind == 2 && second_group_ind == 8)
	//drawEllipse(img,param);
	//组队中的 first group先进行内点准则验证，并且更新组的支持内点数量
	for ( unsigned int i = 0; i<(*groups)[first_group_ind].size(); i++)
	{
		addr = (*groups)[first_group_ind][i] * 8; //第first_group_ind分组的第i条线段索引*8
		rec.x1 = lines[addr];
		rec.y1 = lines[addr+1];
		rec.x2 = lines[addr+2];
		rec.y2 = lines[addr+3];
		rec.x  = (rec.x1 + rec.x2)/2;
		rec.y  = (rec.y1 + rec.y2)/2;
		rec.dx = lines[addr+4];
		rec.dy = lines[addr+5];
		rec.width = 3*distance_tolerance;
		//line_length[i] = (int)lines[addr+6];//记录线段长度到数组line_length[i]
		rec_support_cnt = rec_inliers_cnt = 0;//清零很重要
		if ( lines[addr+7] == 1) //极性一致
		{
			for(iter = ri_ini(&rec);!ri_end(iter);ri_inc(iter))//线段1
			{
				//外接矩形可能会越界
				if(iter->x >= 0 && iter->y >= 0 && iter->x < angles->xsize && iter->y < angles->ysize)
				{
					temp  = angles->data[iter->y*angles->xsize+iter->x] ;//内点的梯度方向
					if(temp!= NOTDEF )
					{
						//test point's normal is (ax0+by0/2+d/2, cy0+bx0/2+e/2)
						point_normalx = Coefficients[0]*iter->x + (Coefficients[1]*iter->y + Coefficients[3])/2;
						point_normaly = Coefficients[2]*iter->y + (Coefficients[1]*iter->x + Coefficients[4])/2;
						point_normal = atan2(-point_normaly,-point_normalx); //边缘点的法线方向,指向椭圆内侧
						rec_inliers_cnt++;
						if(angle_diff(point_normal,temp) <= M_1_8_PI ) //+- 22.5°内 且 || d - r || < 3 dis_t
						{
							rec_support_cnt++;
							pixel_temp.x = iter->x; pixel_temp.y = iter->y;
							first_group_inliers.push_back(pixel_temp);//添加该线段对应的内点
						}
					} 
				}
			}
		}
		else//极性相反
		{
			for(iter = ri_ini(&rec);!ri_end(iter);ri_inc(iter))//线段1
			{
				//外接矩形可能会越界
				if(iter->x >= 0 && iter->y >= 0 && iter->x < angles->xsize && iter->y < angles->ysize)
				{
					temp  = angles->data[iter->y*angles->xsize+iter->x] ;//内点的梯度方向
					if(temp!= NOTDEF )
					{
						//test point's normal is (ax0+by0/2+d/2, cy0+bx0/2+e/2)
						point_normalx = Coefficients[0]*iter->x + (Coefficients[1]*iter->y + Coefficients[3])/2;
						point_normaly = Coefficients[2]*iter->y + (Coefficients[1]*iter->x + Coefficients[4])/2;
						point_normal = atan2(point_normaly,point_normalx); //边缘点的法线方向,指向椭圆外侧
						rec_inliers_cnt++;
						if(angle_diff(point_normal,temp) <= M_1_8_PI ) //+- 22.5°内 且 || d - r || < 3 dis_t
						{
							rec_support_cnt++;
							pixel_temp.x = iter->x; pixel_temp.y = iter->y;
							first_group_inliers.push_back(pixel_temp);//添加该线段对应的内点
						}
					} 
				}
			}
		}
		if( !( rec_support_cnt > 0 && ( rec_support_cnt >= 0.8*lines[addr+6] || rec_support_cnt*1.0/rec_inliers_cnt >= 0.6) ) )
		{
			flag1 = FALSE; //flag1 初始化时为TRUE, 一旦组内有一条线段不满足要求，直接false, 内点准则验证不通过
			break;
		}
	}
	if ( flag1 == TRUE && first_group_inliers.size() >= 0.8*group_inliers_num[first_group_ind] )//靠近最大统计过的内点,通过验证
	{
		if( first_group_inliers.size() >= group_inliers_num[first_group_ind])//更新组出现过的最大内点数
			group_inliers_num[first_group_ind] =  first_group_inliers.size();
	}
	else 
		flag1 = FALSE;
	//第一个组完成验证
	if ( second_group_ind == -1 || fit_matrix2 == NULL)//只对一个覆盖度较大的组进行拟合
	{
		ellipara->x = param[0];//因为无论如何，都需要返回显著性强的椭圆
	    ellipara->y = param[1];
	    ellipara->a = param[2];
	    ellipara->b = param[3];
	    ellipara->phi = param[4];
		if ( flag1 == TRUE)//通过内点再次拟合，提高质量
		{
			point2d * dataxy = (point2d*)malloc(sizeof(point2d)*first_group_inliers.size());
			for ( unsigned int i = 0; i<first_group_inliers.size(); i++)
			{
				dataxy[i].x = first_group_inliers[i].x;
				dataxy[i].y = first_group_inliers[i].y;
			}
			info = fitEllipse(dataxy,first_group_inliers.size(), param2);
			free(dataxy); //释放内存
			if ( info == 1  && isEllipseEqual(param2,param,3*distance_tolerance,0.1,0.1,0.1,0.9) )
			{
				ellipara->x = param2[0];//更新椭圆，提高品质
			    ellipara->y = param2[1];
			    ellipara->a = param2[2];
			    ellipara->b = param2[3];
			    ellipara->phi = param2[4];
			    //drawEllipse(img,param2);
			}
		}
		return TRUE;//对于只有一个组的提取椭圆，此时直接返回
	}
	//接下来，对组队中的 second group进行内点准则验证，并且更新组的支持内点数量
	if (flag1 == FALSE)//在组队运算中，如果第一个组都无法满足内点要求，直接返回false
		return FALSE;
	for ( unsigned int i = 0; i<(*groups)[second_group_ind].size(); i++)
	{
		addr = (*groups)[second_group_ind][i] * 8; //第first_group_ind分组的第i条线段索引*8
		rec.x1 = lines[addr];
		rec.y1 = lines[addr+1];
		rec.x2 = lines[addr+2];
		rec.y2 = lines[addr+3];
		rec.x  = (rec.x1 + rec.x2)/2;
		rec.y  = (rec.y1 + rec.y2)/2;
		rec.dx = lines[addr+4];
		rec.dy = lines[addr+5];
		rec.width = 3*distance_tolerance;
		//line_length[i] = (int)lines[addr+6];//记录线段长度到数组line_length[i]
		rec_support_cnt = rec_inliers_cnt = 0;//清零很重要
		if ( lines[addr+7] == 1) //极性一致
		{
			for(iter = ri_ini(&rec);!ri_end(iter);ri_inc(iter))//线段1
			{
				//外接矩形可能会越界
				if(iter->x >= 0 && iter->y >= 0 && iter->x < angles->xsize && iter->y < angles->ysize)
				{
					temp  = angles->data[iter->y*angles->xsize+iter->x] ;//内点的梯度方向
					if(temp!= NOTDEF )
					{
						//test point's normal is (ax0+by0/2+d/2, cy0+bx0/2+e/2)
						point_normalx = Coefficients[0]*iter->x + (Coefficients[1]*iter->y + Coefficients[3])/2;
						point_normaly = Coefficients[2]*iter->y + (Coefficients[1]*iter->x + Coefficients[4])/2;
						point_normal = atan2(-point_normaly,-point_normalx); //边缘点的法线方向,指向椭圆内侧
						rec_inliers_cnt++;
						if(angle_diff(point_normal,temp) <= M_1_8_PI ) //+- 22.5°内 且 || d - r || < 3 dis_t
						{
							rec_support_cnt++;
							pixel_temp.x = iter->x; pixel_temp.y = iter->y;
							second_group_inliers.push_back(pixel_temp);//添加该线段对应的内点
						}
					} 
				}
			}
		}
		else//极性相反
		{
			for(iter = ri_ini(&rec);!ri_end(iter);ri_inc(iter))//线段1
			{
				//外接矩形可能会越界
				if(iter->x >= 0 && iter->y >= 0 && iter->x < angles->xsize && iter->y < angles->ysize)
				{
					temp  = angles->data[iter->y*angles->xsize+iter->x] ;//内点的梯度方向
					if(temp!= NOTDEF )
					{
						//test point's normal is (ax0+by0/2+d/2, cy0+bx0/2+e/2)
						point_normalx = Coefficients[0]*iter->x + (Coefficients[1]*iter->y + Coefficients[3])/2;
						point_normaly = Coefficients[2]*iter->y + (Coefficients[1]*iter->x + Coefficients[4])/2;
						point_normal = atan2(point_normaly,point_normalx); //边缘点的法线方向,指向椭圆外侧
						rec_inliers_cnt++;
						if(angle_diff(point_normal,temp) <= M_1_8_PI ) //+- 22.5°内 且 || d - r || < 3 dis_t
						{
							rec_support_cnt++;
							pixel_temp.x = iter->x; pixel_temp.y = iter->y;
							second_group_inliers.push_back(pixel_temp);//添加该线段对应的内点
						}
					} 
				}
			}
		}
		if( !(rec_support_cnt > 0 && ( rec_support_cnt >= 0.8*lines[addr+6] || rec_support_cnt*1.0/rec_inliers_cnt >= 0.6) ) )
		{
			flag2 = FALSE; //flag1 初始化时为TRUE, 一旦组内有一条线段不满足要求，直接false, 内点准则验证不通过
			break;
		}
	}
	if ( flag2 == TRUE && second_group_inliers.size() >= 0.8*group_inliers_num[second_group_ind] )//靠近最大统计过的内点,通过验证
	{
		if(second_group_inliers.size() >= group_inliers_num[second_group_ind])//更新组出现过的最大内点数
			group_inliers_num[second_group_ind] = second_group_inliers.size();
	}
	else 
		flag2 = FALSE;
	//第二个组完成验证
	if ( flag1 == TRUE && flag2 == TRUE)
	{
		point2d * dataxy = (point2d*)malloc(sizeof(point2d)*(first_group_inliers.size() + second_group_inliers.size()));
		for ( unsigned int i = 0; i<first_group_inliers.size(); i++)
		{
			dataxy[i].x = first_group_inliers[i].x;
			dataxy[i].y = first_group_inliers[i].y;
		}
		addr = first_group_inliers.size();
		for ( unsigned int i = 0; i<second_group_inliers.size(); i++)//连接两个数组时一定要注意索引范围
		{
			dataxy[addr+i].x = second_group_inliers[i].x;
			dataxy[addr+i].y = second_group_inliers[i].y;
		}
//		drawEdge(img,dataxy,(first_group_inliers.size() + second_group_inliers.size()));
		info = fitEllipse(dataxy,(first_group_inliers.size() + second_group_inliers.size()), param2);
		free(dataxy); //释放内存
		//小长短轴的椭圆需要放宽参数
		if ( param[2] <= 50 )
			semimajor_errorratio = 0.25;
		else if (param[2] <= 100 )
			semimajor_errorratio = 0.15;
		else
			semimajor_errorratio = 0.1;
		if ( param[3] <= 50 )
			semiminor_errorratio = 0.25;
		else if ( param[3] <= 100)
			semiminor_errorratio = 0.15;
		else
			semiminor_errorratio = 0.1;
		if (param[2] <= 50 && param[3] <= 50 )
			iscircle_ratio = 0.75;
		else if (param[2] >= 50 && param[2] <= 100 &&  param[3] >= 50 && param[3] <= 100 )
			iscircle_ratio = 0.85;
		else
			iscircle_ratio = 0.9;
		if ( info == 1  && isEllipseEqual(param2,param,3*distance_tolerance,semimajor_errorratio,semiminor_errorratio,0.1, iscircle_ratio) )
		{
			ellipara->x = param2[0];//更新椭圆，提高品质
		    ellipara->y = param2[1];
		    ellipara->a = param2[2];
		    ellipara->b = param2[3];
		    ellipara->phi = param2[4];
		    //drawEllipse(img,param2);
			return TRUE;
		}
	}
	return FALSE;
}


//输入
//lsd算法检测得到的线段集合lines的数量line_num，return的返回值是line_nums条线段，为一维double型数组lines，长度为8*n，每8个为一组
//存着x1,y1,x2,y2,dx,dy,length,polarity
//groups: 线段分组，每个组存按照几何分布顺序顺时针或者逆时针存储着线段索引，线段索引范围是0~line_num-1
//coverages: 每个分组的角度覆盖范围0~2pi，如果组里只有1条线段，覆盖角度为0。数组长度等于分组的数量。
//angles 存边缘点的梯度方向gradient direction, 无边缘点位NOTDEF
//返回值 PairedGroupList* list 返回的是初始椭圆集合的数组，长度list->length. 
//切记，该内存在函数内申请，用完该函数记得释放内存，调用函数freePairedSegmentList()进行释放

PairGroupList * getValidInitialEllipseSet( double * lines, int line_num, vector<vector<int>> * groups, double * coverages, image_double angles, double distance_tolerance, int specified_polarity)
{
	//加速计算
	//int* lineInliersIndex = (int*)malloc(sizeof(int)*line_num);//如果第i条线段找到了内点，则记录其索引为j = length(supportInliers),即supportInliers.at(j)存着该线段的支持内点,没找到内点的线段对应索引为初始值-1.
    //vector<vector<point2d>> supportInliers;//保存相应线段的支持内点
	//memset(lineInliersIndex,-1,sizeof(int)*line_num);//此处要实践确实可行，对于整数可以初始化为0，-1.对于浮点数则只可以为0.

	PairGroupList * pairGroupList = NULL;
	PairGroupNode *head, *tail;
	int pairlength = 0;
	point2d pointG1s,pointG1e,pointG2s,pointG2e,g1s_ls_dir,g1e_ls_dir,g2s_ls_dir,g2e_ls_dir;
	double polarity;
	point5d ellipara;
    int groupsNum = (*groups).size();//组的数量
	double * fitMatrixes = (double*)malloc(sizeof(double)*groupsNum*36);//定义拟合矩阵S_{6 x 6}. 每个组都有一个拟合矩阵
	unsigned int * supportInliersNum = (unsigned int*)malloc(sizeof(int)*groupsNum);//用于存储每个组曾经最大出现的支持内点数量
	memset(fitMatrixes,0,sizeof(double)*groupsNum*36);
	memset(supportInliersNum, 0, sizeof(unsigned int)*groupsNum);//初始化为0.
	//double distance_tolerance = max( 2.0, 0.005*min(angles->xsize,angles->ysize) ); // 0.005%*min(xsize,ysize)
    int i,j;
	int cnt_temp,ind_start,ind_end;
	bool info;
    
	//实例化拟合矩阵Si
	point2d * dataxy = (point2d*)malloc(sizeof(point2d)*line_num*2);//申请足够大内存, line_num条线段，共有2line_num个端点
	for ( i = 0; i<groupsNum; i++)
	{
		cnt_temp = 0;//千万注意要清0
		for ( j = 0; j<(*groups)[i].size(); j++)
		{
			//每一条线段有2个端点
			dataxy[cnt_temp].x = lines[(*groups)[i][j]*8];
			dataxy[cnt_temp++].y = lines[(*groups)[i][j]*8+1];
			dataxy[cnt_temp].x = lines[(*groups)[i][j]*8+2];
			dataxy[cnt_temp++].y = lines[(*groups)[i][j]*8+3];
		}
		calcuFitMatrix(dataxy,cnt_temp, fitMatrixes+i*36);
	}
	free(dataxy);//释放内存

	head = tail = NULL;//将初始椭圆集合存储到链表中
	//selection of salient elliptic hypothesis
	for ( i = 0; i<groupsNum; i++)
	{
		if(coverages[i] >= M_4_9_PI )//当组的覆盖角度>= 4pi/9 = 80°, 我们认为具有很大的显著性，可直接拟合提取
		{
			//加入极性判断,只提取指定极性的椭圆
			if (specified_polarity == 0 || (lines[(*groups)[i][0]*8+7] == specified_polarity))
			{
				//显著性大的初始椭圆提取，一定会返回TRUE，因此没必要再判断
				info = calcEllipseParametersAndValidate(lines,line_num,groups,i,-1,(fitMatrixes+i*36),NULL,angles,distance_tolerance,supportInliersNum,&ellipara);
				if (info == FALSE) 
				{
					continue;
					error("getValidInitialEllipseSet, selection of salient ellipses failed!");//这种情况会出现？？,跑54.jpg出现该问题
				}
				PairGroupNode * node = (PairGroupNode*)malloc(sizeof(PairGroupNode));
				node->center.x = ellipara.x;
				node->center.y = ellipara.y;
				node->axis.x   = ellipara.a;
				node->axis.y   = ellipara.b;
				node->phi      = ellipara.phi;
				node->pairGroupInd.x = i;
				node->pairGroupInd.y = -1;//无
				if(head != NULL)
				{
					tail->next = node;
					tail = node;
				}
				else
				{
					head = tail = node;
				}
				pairlength++;
			}
		}
	}
    //selection of pair group hypothesis
	for ( i = 0; i<groupsNum-1; i++)
		for ( j = i+1; j<groupsNum; j++)
			{
				//加入极性判断,只提取指定极性的椭圆
			   if (specified_polarity == 0 || (lines[(*groups)[i][0]*8+7] == specified_polarity))
			    {
					//group i 's polarity is the same as group j; and the number of two paired groups should be >= 3.
					if( lines[(*groups)[i][0]*8+7] == lines[(*groups)[j][0]*8+7] && ((*groups)[i].size() + (*groups)[j].size()) >= 3)
					{
						ind_start = (*groups)[i][0];//第i组的最开始一条线段索引
						ind_end   = (*groups)[i][(*groups)[i].size()-1];//第i组的最后一条线段索引
						pointG1s.x = lines[ind_start*8];
						pointG1s.y = lines[ind_start*8+1];
						g1s_ls_dir.x = lines[ind_start*8+4];
						g1s_ls_dir.y = lines[ind_start*8+5];
						pointG1e.x = lines[ind_end*8+2];
						pointG1e.y = lines[ind_end*8+3];
						g1e_ls_dir.x = lines[ind_end*8+4];
						g1e_ls_dir.y = lines[ind_end*8+5];
						ind_start = (*groups)[j][0];//第j组的最开始一条线段索引
						ind_end   = (*groups)[j][(*groups)[j].size()-1];//第j组的最后一条线段索引
						pointG2s.x = lines[ind_start*8];
						pointG2s.y = lines[ind_start*8+1];
						g2s_ls_dir.x = lines[ind_start*8+4];
						g2s_ls_dir.y = lines[ind_start*8+5];
						pointG2e.x = lines[ind_end*8+2];
						pointG2e.y = lines[ind_end*8+3];
						g2e_ls_dir.x = lines[ind_end*8+4];
						g2e_ls_dir.y = lines[ind_end*8+5];
						polarity = lines[ind_start*8+7]; //i,j两组的极性
						if(regionLimitation(pointG1s,g1s_ls_dir,pointG1e,g1e_ls_dir,pointG2s,g2s_ls_dir,pointG2e,g2e_ls_dir,polarity,-3*distance_tolerance))//都在彼此的线性区域内
						{
							//if ( i == 2)
							//	drawPairGroup(img,lines,(*groups),i,j);

							if(calcEllipseParametersAndValidate(lines,line_num,groups,i,j,(fitMatrixes+i*36),(fitMatrixes+j*36),angles,distance_tolerance,supportInliersNum,&ellipara))//二次一般方程线性求解，线段的内点支持比例
							{
								PairGroupNode * node = (PairGroupNode*)malloc(sizeof(PairGroupNode));
								node->center.x = ellipara.x;
								node->center.y = ellipara.y;
								node->axis.x   = ellipara.a;
								node->axis.y   = ellipara.b;
								node->phi      = ellipara.phi;
								node->pairGroupInd.x = i;
								node->pairGroupInd.y = -1;//无
								if(head != NULL)
								{
									tail->next = node;
									tail = node;
								}
								else
								{
									head = tail = node;
								}
								pairlength++;
							}
						}
						
					}
			   }
			}
	if(pairlength > 0)
	{
		PairGroupNode *p;
		p = head;
		pairGroupList = pairGroupListInit(pairlength);
		for( int i = 0; i<pairGroupList->length; i++)
		{
			pairGroupList->pairGroup[i].center.x = p->center.x;
			pairGroupList->pairGroup[i].center.y = p->center.y;
			pairGroupList->pairGroup[i].axis.x = p->axis.x;
			pairGroupList->pairGroup[i].axis.y = p->axis.y;
			pairGroupList->pairGroup[i].phi = p->phi;
			pairGroupList->pairGroup[i].pairGroupInd.x = p->pairGroupInd.x;//记录组对(i,j),由groups中的第i个组和第j个组构成的匹配组产生该有效椭圆参数
			pairGroupList->pairGroup[i].pairGroupInd.y = p->pairGroupInd.y;
			p = p->next;
		}
		tail->next = NULL;
		while (head != NULL)
		{
			p = head;
			head = head->next;
			free(p);
		}
	}
	//supportInliers.resize(0);
	//free(lineInliersIndex);//释放线段内点的索引
	free(supportInliersNum);//释放存储各个组的支持内点数量的数组
	free(fitMatrixes);//释放存储各个组的拟合矩阵
	return pairGroupList;
}


void generateEllipseCandidates( PairGroupList * pairGroupList, double distance_tolerance, double * & ellipse_candidates, int * candidates_num)
{
	if( pairGroupList->length <= 0 )//检测，至少要有1个样本用来产生候选
	{
		ellipse_candidates = NULL;
		(*candidates_num) = 0;
		return;
	}
	double * centers;
	int center_num; //椭圆中心(xi,yi)的聚类数量
	double * phis;
	int phi_num;    //针对每一个椭圆中心(xi,yi)，倾斜角度phi的聚类数量
	double * axises;
	int axis_num;   //针对每一个椭圆中心和倾角(xi,yi,phi),长短半轴(a,b)的聚类数量
	double * bufferXY = (double*)calloc(pairGroupList->length*2,sizeof(double));
	double * bufferPhi = (double*)calloc(pairGroupList->length,sizeof(double));
	double * bufferAB = (double*)calloc(pairGroupList->length*2,sizeof(double));
	point2i * bufferIndexes = (point2i *)calloc(pairGroupList->length,sizeof(point2i));//point[i].x记录第i个分类在bufferXX中的起始索引位置，point[i].y记录第i个分类在bufferXX中的长度
	double  * buffer2AB = (double*)calloc(pairGroupList->length*2,sizeof(double));
	point2i * buffer2Indexes = (point2i *)calloc(pairGroupList->length,sizeof(point2i));//point[i].x记录第i个分类在bufferXX中的起始索引位置，point[i].y记录第i个分类在bufferXX中的长度
	int     * buffer_temp = (int*)calloc(pairGroupList->length,sizeof(int));
	int addr,addr2,info,ind;
	double dis_min,dis_temp;
	if ( bufferXY == NULL || bufferPhi == NULL || bufferAB == NULL || bufferIndexes == NULL ||
		 buffer2AB == NULL || buffer2Indexes == NULL || buffer_temp == NULL
		)
	{
		ellipse_candidates = NULL;
		(*candidates_num) = 0;
		error("generateEllipseCandidates, not enough memory");
	}
	(*candidates_num) = 0; //候选椭圆数量，初始化为0,非常重要
	//copy
	for ( int i = 0; i<pairGroupList->length; i++)
	{
		addr = 2*i;
		bufferXY[addr] = pairGroupList->pairGroup[i].center.x;
		bufferXY[addr+1] = pairGroupList->pairGroup[i].center.y;
	}
	//cluster the ellipses' centers
	info = cluster2DPoints(bufferXY,pairGroupList->length,distance_tolerance,centers,&center_num);
	if( info == 0)
	{
		ellipse_candidates = NULL;
		(*candidates_num) = 0;
		error("generateEllipseCandidates, cluster2DPoints, error in clustering elliptic centers");
	}
	//classification,寻找每个点归属的聚类中心
	for ( int i = 0; i<pairGroupList->length; i++)
	{
		dis_min = DBL_MAX;
		ind = -1;
		for ( int j = 0; j<center_num; j++)
		{
			addr = 2*j;
			dis_temp = (pairGroupList->pairGroup[i].center.x - centers[addr])*(pairGroupList->pairGroup[i].center.x - centers[addr]) + (pairGroupList->pairGroup[i].center.y - centers[addr+1])*(pairGroupList->pairGroup[i].center.y - centers[addr+1]);
			if(dis_temp < dis_min)
			{
				dis_min = dis_temp;
				ind = j; //record the nearest center's index
			}
		}
		buffer_temp[i] = ind; //此处借用buffer2来记下第i个初始椭圆对应第ind个椭圆聚类中心
	}
	//将分类结果按顺序存到bufferXY,bufferPhi,bufferAB中，且bufferIndexes[i]存着第i个聚类中心的起始索引位置和长度
	memset(bufferIndexes,0,sizeof(point2i)*pairGroupList->length);
	ind = 0;//清零，样本点起始位置，索引位置是ind*2,分区的基址
	for ( int i = 0; i<center_num; i++)
	{
		bufferIndexes[i].x = ind; 
		for ( int j = 0; j<pairGroupList->length; j++)
		{
			if ( buffer_temp[j] == i)
			{
				addr = ind*2;//切记长短半轴是一组一组寸储的，需要 x 2
				addr2 = bufferIndexes[i].y*2;
				bufferPhi[ind+bufferIndexes[i].y] = pairGroupList->pairGroup[j].phi;
				bufferAB[addr+addr2] = pairGroupList->pairGroup[j].axis.x;
				bufferAB[addr+addr2+1] = pairGroupList->pairGroup[j].axis.y;
				bufferIndexes[i].y++;//第i个聚类中心周围的点数量加1
			}
		}
		if(bufferIndexes[i].y == 0)//聚类中心周围没有靠近的点
		{
			error("generateEllipseCandidates, no XY points near to the clustering center");
		}
		ind += bufferIndexes[i].y;
	}
	//cout<<"2D cluster centers over"<<endl;
	//对每一个椭圆中心的周围的点进行倾角聚类
	//第i个椭圆聚类中心，其邻近点的索引范围是：bufferIndexs[i].x ~ (bufferIndex[i].x + bufferIndex[i].y-1)
	for ( int i = 0; i<center_num; i++)
	{
		

		double * phi_pointer_temp = bufferPhi+bufferIndexes[i].x;//倾角指针
		double * ab_pointer_temp = bufferAB+bufferIndexes[i].x*2;//长短半轴的指针,记住 x 2
		info = cluster1DDatas(phi_pointer_temp, bufferIndexes[i].y, 0.0873, phis, &phi_num);//对phi聚类, pi/180*5 = 0.0873, 5°误差
		if (info == 0) //不懂为什么，聚类中心centers[i]的周围可能没有最靠近它的点,数量bufferIndexes[i].y = 0
		{ 
			//cout<<"generateEllipseCandidates, cluster2DPoints, error in clustering elliptic phis"<<endl;
			continue;
			//error("generateEllipseCandidates, cluster2DPoints, error in clustering elliptic phis");
		}
		//classification,寻找每个点归属的聚类中心
		for ( int j = 0; j<bufferIndexes[i].y; j++ )
		{
			dis_min = DBL_MAX;
			ind = -1;
			for ( int k = 0; k<phi_num; k++)
			{
				dis_temp = (*(phi_pointer_temp+j)-phis[k]) * (*(phi_pointer_temp+j)-phis[k]);
				if(dis_temp < dis_min)
				{
					dis_min = dis_temp;
					ind = k;//record the nearest phi's index
				}
			}
			buffer_temp[j] = ind;
		}
		//将分类结果按顺序存储到buffer2AB中，且buffer2Indexes[j].x对应第i个phi的聚类中心起始点，buffer2Indexes[j].y对应数量(长度)
		memset(buffer2Indexes,0,sizeof(point2i)*bufferIndexes[i].y);
		ind = 0;
		for ( int j = 0; j<phi_num; j++)
		{
			buffer2Indexes[j].x = ind;//起始点
			for ( int k = 0; k<bufferIndexes[i].y; k++)
			{
				if ( buffer_temp[k] == j)
				{
					addr = ind*2;
					addr2 = buffer2Indexes[j].y*2;
					buffer2AB[addr+addr2] = *(ab_pointer_temp+k*2);
					buffer2AB[addr+addr2+1] = *(ab_pointer_temp+k*2+1);
					buffer2Indexes[j].y++;//长度加1
				}
			}
			ind += buffer2Indexes[j].y;
		}
		for ( int j = 0; j<phi_num; j++ )
		{
			double * ab_pointer_temp2 = buffer2AB+buffer2Indexes[j].x*2; //长短半轴的指针,记住 x 2
			info = cluster2DPoints(ab_pointer_temp2, buffer2Indexes[j].y, distance_tolerance, axises, &axis_num);
			if (info == 0) //不懂为什么，聚类中心phi_j的周围可能没有最靠近它的点,数量buffer2Indexes[j].y = 0
			{   
				//cout<<"generateEllipseCandidates, cluster2DPoints, error in clustering elliptic axises"<<endl;
				continue;
				//error("generateEllipseCandidates, cluster2DPoints, error in clustering elliptic axises");
			}
			//将候选椭圆重写到bufferXY,bufferPhi,bufferAB里面, 候选椭圆数量(*candidates_num)++
			for ( int k = 0; k<axis_num; k++)
			{
				addr = (*candidates_num)*2;
				bufferXY[addr] = centers[i*2];
				bufferXY[addr+1] = centers[i*2+1];
				bufferPhi[(*candidates_num)] = phis[j];
				bufferAB[addr] = axises[k*2];
				bufferAB[addr+1] = axises[k*2+1];
				(*candidates_num)++;
			}
			free(axises);//cluster2DPoints严格要求，用完axises后，需要释放函数内部申请的内存
		}
		free(phis);//cluster1DDatas严格要求，用完phis后，需要释放函数内部申请的内存
	}
	free(centers);//cluster2DPoints严格要求，用完centers后，需要释放函数内部申请的内存
	//释放在函数开头申请的部分内存
	free(buffer_temp); //此处释放出问题
	free(buffer2Indexes);
	free(buffer2AB);
	free(bufferIndexes);
	ellipse_candidates = (double*)malloc(sizeof(double)*(*candidates_num)*5);
	for ( int i = 0; i < (*candidates_num); i++ )
	{
		addr = 2*i;
		ellipse_candidates[i*5]  = bufferXY[addr];
		ellipse_candidates[i*5+1]= bufferXY[addr+1];
		ellipse_candidates[i*5+2]= bufferAB[addr];
		ellipse_candidates[i*5+3]= bufferAB[addr+1];
		ellipse_candidates[i*5+4]= bufferPhi[i];
	}
	//释放在函数开头申请的内存
	free(bufferAB);
	free(bufferPhi);
	free(bufferXY);
	if((*candidates_num)<= 0)
	{
		*candidates_num = 0;
		ellipse_candidates = NULL;
		//cout<<"no any candidates generated!"<<endl;
	}
}







//==========================================END=======================================================================
/**
输入：
prhs[0]: 输入的灰度图像，单通道，大小是imgy x imgx
prhs[1]: 边缘提取选择，1 canny; 2 sobel
prhs[2]: 检测指定的椭圆极性
输出：
plhs[0]: 候选椭圆组合(xi,yi,ai,bi,phi_i)', 5 x m
plhs[1]: 边缘图，大小是imgy x imgx，设边缘点总数为 edgepix_n. 二值化，0 或者 255
plhs[2]: 边缘点的梯度向量矩阵，大小是 2 x edgepix_n, (cos(theta_rad),sin(theta_rad))'...
plhs[3]: 线段图，大小是imgy x imgx 
*/
/*
compile：
mex generateEllipseCandidates.cpp -IF:\OpenCV\opencv2.4.9\build\include -IF:\OpenCV\opencv2.4.9\build\include\opencv -IF:\OpenCV\opencv2.4.9\build\include\opencv2 -LF:\OpenCV\opencv2.4.9\build\x64\vc11\lib -IF:\Matlab\settlein\extern\include -LF:\Matlab\settlein\extern\lib\win64\microsoft -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249 -llibmwlapack.lib
*/
//======================================MEX function==================================================================

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs!=3) 
      mexErrMsgIdAndTxt( "MATLAB:revord:invalidNumInputs","One input required.");
    else if(nlhs > 4) 
      mexErrMsgIdAndTxt( "MATLAB:revord:maxlhs","Too many output arguments.");
	uchar * inputimg = (uchar*)mxGetData(prhs[0]);
	int imgy,imgx;
	int edge_process_select = (int)mxGetScalar(prhs[1]);//边缘提取选择，1 canny; 2 sobel
	int specified_polarity  = (int)mxGetScalar(prhs[2]);//1,指定检测的椭圆极性要为正; -1指定极性为负; 0表示两种极性椭圆都检测
	imgy = (int)mxGetM(prhs[0]);
	imgx = (int)mxGetN(prhs[0]);
	double *data=(double*)malloc(imgy*imgx*sizeof(double));//将输入矩阵中的图像数据转存到一维数组中
    for(int c=0;c<imgx;c++)
    {
        for(int r=0;r<imgy;r++)
        {
           data[c+r*imgx]=inputimg[r+c*imgy];              
        }    
    }
	int n;//线段数量
	//int new_n;
	vector<vector<int>> groups;
	double * coverages;
	int * reg;
	int reg_x;
	int reg_y;
    double* out=mylsd(&n, data,imgx,imgy,&reg,&reg_x,&reg_y);
	groupLSs(out,n,reg,reg_x,reg_y,&groups);//分组
	free(reg); //释放内存
	calcuGroupCoverage(out,n,groups,coverages);//计算每个组的覆盖角度

    printf("The number of output arc-support line segments: %i\n",n);
	printf("The number of arc-support groups:%i\n",groups.size());
	/*int groups_t = 0;
	for (int i = 0; i<groups.size(); i++)
	{ 
		groups_t+= groups[i].size();
	}
	printf("Groups' total ls num:%i\n",groups_t);*/

	 image_double angles;
	 if(edge_process_select == 1)
		calculateGradient2(data,imgx,imgy,&angles); //version2, sobel; version 3 canny
	 else 
		 calculateGradient3(data,imgx,imgy,&angles); //version2, sobel; version 3 canny
	 PairGroupList * pairGroupList;
	 double distance_tolerance = 2;//max( 2.0, 0.005*min(angles->xsize,angles->ysize) ); // 0.005%*min(xsize,ysize)
	 double * candidates; //候选椭圆
	 double * candidates_out;//输出候选椭圆指针
	 int  candidates_num = 0;//候选椭圆数量
	 //rejectShortLines(out,n,&new_n);
	 pairGroupList = getValidInitialEllipseSet(out,n,&groups,coverages,angles,distance_tolerance,specified_polarity);
	 if(pairGroupList != NULL)
	 {
		printf("The number of initial ellipses：%i \n",pairGroupList->length);
		generateEllipseCandidates(pairGroupList, distance_tolerance, candidates, &candidates_num);
		printf("The number of ellipse candidates: %i \n",candidates_num);
		
		plhs[0] = mxCreateDoubleMatrix(5,candidates_num,mxREAL);
		candidates_out = (double*)mxGetPr(plhs[0]);
		//候选圆组合(xi,yi,ai,bi,phi_i)', 5 x candidates_num, 复制到矩阵candidates_out中
		memcpy(candidates_out,candidates,sizeof(double)*5*candidates_num);

		freePairGroupList(pairGroupList);
		free(candidates);
	 }
	 else
	 {
		 printf("The number of initial ellipses：%i \n",0);
		 double *candidates_out;
		 plhs[0] = mxCreateDoubleMatrix(5,1,mxREAL);
		 candidates_out = (double*)mxGetPr(plhs[0]);
		 candidates_out[0] = candidates_out[1] = candidates_out[2] = candidates_out[3] = candidates_out[4] = 0;
	 }
	 uchar *edgeimg_out;
	 unsigned long edge_pixels_total_num = 0;//边缘总像素
	 double *gradient_vec_out;
	 plhs[1] = mxCreateNumericMatrix(imgy,imgx,mxUINT8_CLASS,mxREAL);
	 edgeimg_out = (uchar*)mxGetData(plhs[1]);
	 //将边缘图复制到矩阵edgeimg_out中
	 //将梯度向量存到矩阵gradient_vec_out中
	 unsigned long addr,g_cnt = 0;
	 for ( int c = 0; c < imgx; c++ )
		 for ( int r = 0; r < imgy; r++)
		 {
			 addr = r*imgx+c;
			 if(angles->data[addr] == NOTDEF)
				 edgeimg_out[c*imgy+r] = 0;
			 else
			 {
				 edgeimg_out[c*imgy+r] = 255;//为边缘点，赋值为白色
				 //------------------------------------------------
				 edge_pixels_total_num++;
			 }
		 }
	 printf("edge pixel number: %i\n",edge_pixels_total_num);
	//申请edge_pixels_total_num x 2 来保存每一个边缘点的梯度向量，以列为优先，符合matlab的习惯
	 plhs[2] = mxCreateDoubleMatrix(2,edge_pixels_total_num,mxREAL);
	 gradient_vec_out = (double*)mxGetPr(plhs[2]);
	  for ( int c = 0; c < imgx; c++ )
		 for ( int r = 0; r < imgy; r++)
		 {
			 addr = r*imgx+c;
			 if(angles->data[addr] != NOTDEF)
			 {
				 gradient_vec_out[g_cnt++] = cos(angles->data[addr]);
				 gradient_vec_out[g_cnt++] = sin(angles->data[addr]);
			 }
		 }
	 //---------------------------------------------------------------------
	//输出线段检测的图像
	if(nlhs == 4)
	{
		Mat ls_mat = Mat::zeros(imgy,imgx,CV_8UC1);
		for ( int i = 0; i<n ; i++)//draw lines
		{
		  Point2d p1(out[8*i],out[8*i+1]),p2(out[8*i+2],out[8*i+3]);
		  line(ls_mat,p1,p2,Scalar(255,0,0));
		}
		if(candidates_num > 0)//draw ellipses
		{
			for ( int i = 0; i<candidates_num; i++)
				ellipse(ls_mat,cv::Point((int)candidates_out[i*5],(int)candidates_out[i*5+1]),cv::Size(candidates_out[i*5+2],candidates_out[i*5+3]),candidates_out[i*5+4]*180/M_PI,0,360,(Scalar(255,0,0)),1);
		}
		plhs[3] = mxCreateDoubleMatrix(imgy,imgx,mxREAL);
		double * ls_img_out = (double*)mxGetPr(plhs[3]);
		//memcpy(ls_out_mat,ls_mat.data ,sizeof(unsigned char)*M*N);
		for (int i = 0; i<imgx; i++)
			for (int j = 0; j<imgy;j++)
				ls_img_out[i*imgy+j]=ls_mat.data[j*imgx+i];
	}
	//---------------------------------------------------------------------
	//这里的free是释放程序中用于产生候选圆所用到的一系列内存
	free(data);
	free(coverages);
	free(out);
	free_image_double(angles);

}










/*
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
	int m = mxGetM(prhs[0]);
	int n = mxGetN(prhs[0]);
	double * p = (double*)mxGetData(prhs[0]);
	int sum = 0;
	for (int c = 0; c<n; c++)
		for ( int r = 0; r<m; r++)
			sum += p[c*m+r];
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	double *pout = mxGetPr(plhs[0]);
	*pout = sum;

}
*/