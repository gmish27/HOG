#include "mypeopledetect.h"

myMat *readPGM(const char *filename)
{
    char buff[16];
    myMat *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    //read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '5') {
         fprintf(stderr, "Invalid PGM image format (must be 'P5')\n");
         exit(1);
    }

    //alloc memory form image
    img = (myMat*)malloc(sizeof(myMat));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }
	img->type = myCV_8U;
	img->dims = 2;
	img->channels = 1;
	img->step = img->channels*img->cols;
	img->totalsize = (img->step*img->rows);
    while (fgetc(fp) != '\n') ;
	img->data = (uchar*)malloc(img->totalsize);
	img->dataend = img->datalimit = (uchar*)img->data + img->totalsize;

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }
    //read pixel data from file
    if (fread(img->data, img->channels * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }    

    fclose(fp);
    return img;
}


int main(int argc, char** argv)
{
	myMat *img;
	char str[13];
	int i;
	clock_t t;
	hog_ hog;
	rect_ *found=NULL, *found_filtered=NULL;
	size_ w,p;
    img = readPGM(argv[1]);
	strncpy(str,&(argv[1])[5],13);

	hogalc(&hog);        
	
	sizealc(&w, 8, 8);
	sizealc(&p, 32, 32);
	t=clock();
    detectMultiScale(&hog, img, &found, 0, w, p, 1.05, 2, 0);
	t=clock()-t;
	printf("\nDetection Time= %fsecs\n", ((float)t)/CLOCKS_PER_SEC); 	
	if(sbcount(found))
		for (i=0; i<sbcount(found); i++)
			printf("%s x=%d y=%d w=%d h=%d\n",str,found[i].x,found[i].y, found[i].width,found[i].height);
	else
		printf("%s x=%d y=%d w=%d h=%d\n",str,0,0,0,0);
	sbfree(found);
	sbfree(found_filtered);
	free(img->data);
	free(img);
	return 0;
}


