#include <iostream>
#include <fstream>
#include <map>
#include <stdlib.h> //atoi

#include <stdio.h>
#include <string>

using std::endl;
using std::cout;

using namespace std;

int main(int argc, char *argv[]){

///////////////////////////////////////////////////           FILE         ////////////////////////////////////////////////////////////////

	int weights;  // weights // in future will be used

	std::ifstream file(argv[1]); // first command-line argument is taken as name of input //construct calls the open file.
	int nonzeros;
  string s_tmp;

  while(!file.eof()) {
  	getline(file, s_tmp);
  	nonzeros ++;
  }

	cout << "Total number of vertices " << nonzeros << endl;

	file.clear();
  file.seekg(0, ios::beg);
	int *vertices = new int[nonzeros*2];

	if(file.is_open()){ //read if file is openable

           for(int i = 0; i < nonzeros; i++) //read up to number nonzeros
              {
                file >> vertices[i] >> vertices[i+nonzeros] ; // first element is for candidates and second one is for vertices.
              }

        }

	cout << "data was read " << endl;

	//for(int i = 0  ;i < nonzeros*2 ; i++)
   	//cout << " " << vertices[i]  << endl;



/////////////////////////////////////////////////////////////MAPPING///////////////////////////////////////////////////////////////
   int *vertices_mapped = new int[nonzeros*2];
   map<int , int> m;

   //m[vertices[0]] = 0;
   int count = 0;

   for(int i = 0; i < nonzeros*2; i++){
	 if(m[vertices[i]] == 0)
	   m[vertices[i]] = count++;
   }


   for(int i = 0; i < nonzeros*2 ; i++){
		 vertices_mapped[i] = m[vertices[i]];
		 //cout << vertices[i] << " " << vertices_mapped[i] - 1 << endl;
   }
   int vertices_count = count - 1;
 ///////////////////////////////////////////////////////////MAPPING///////////////////////////////////////////////////////////////

 free(vertices);
	FILE *file2;

	if((file2=fopen("mapped_undirected_graph_new.txt", "wb"))==NULL)
	{
	    printf("Something went wrong reading %s\n", "test.txt");
	    return 0;
	}
	else
	{
		 //fprintf(file2, "%d %d %zu\n",vertices_count,vertices_count,nonzeros);
 		 for(int i = 0; i < nonzeros; i++)
       	 	   fprintf(file2, "%d %d\n",vertices_mapped[i] - 1,vertices_mapped[i+nonzeros] - 1);
		   //printf("%d %d \n",vertices[i],vertices_mapped[i]);
	}
	fclose(file2);

	FILE *file3= fopen("file.bin", "wb");
	fwrite(vertices_mapped, sizeof(int), 2*nonzeros, file3);
	fclose(file3);
	cout << "conversion operation is done and saved as file.bin" << endl;

  free(vertices_mapped);

}
