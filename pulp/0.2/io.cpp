
void read_adj(char* filename, int& n, long& m,
  int*& out_array, long*& out_degree_list,
  bool has_vert_weights, bool has_edge_weights,
  int*& vertex_weights, int*& edge_weights, long& vertex_weights_sum)
{
  ifstream infile;
  string line;
  string val;

  out_array = new int[m];
  out_degree_list = new long[n+1];
  if (has_vert_weights || has_edge_weights) vertex_weights = new int[n];
  else vertex_weights = NULL;
  if (has_edge_weights || has_vert_weights) edge_weights = new int[m];
  else edge_weights = NULL;
  vertex_weights_sum = 0;

#pragma omp parallel for
  for (int i = 0; i < n+1; ++i)
    out_degree_list[i] = 0;

  long count = 0;
  int cur_vert = 0;

  infile.open(filename);
  getline(infile, line);

  while (getline(infile, line))
  {
    stringstream ss(line);
    out_degree_list[cur_vert] = count;
    if (has_vert_weights)
    {
      getline(ss, val, ' ');
      vertex_weights[cur_vert] = atoi(val.c_str());
      vertex_weights_sum += vertex_weights[cur_vert];
    }
    else if (has_edge_weights)
    {
      vertex_weights[cur_vert] = 1;
      vertex_weights_sum += vertex_weights[cur_vert];
    }
    /*else
    {
      vertex_weights[cur_vert] = rand() % 10;
      vertex_weights_sum += vertex_weights[cur_vert];
    }*/
    ++cur_vert;

    while (getline(ss, val, ' '))
    {
      out_array[count] = atoi(val.c_str())-1;
      if (has_edge_weights)
      {
        getline(ss, val, ' ');
        edge_weights[count] = atoi(val.c_str());
      }
      else if (has_vert_weights)
      {
        edge_weights[count] = 1;
      }
      /*else
      {
        edge_weights[count] = rand() % 10;
      }*/
      ++count;
    }
  }
  out_degree_list[cur_vert] = count;
  assert(cur_vert == n);
  assert(count == m);

  infile.close();  
}

void read_graph(char* filename, int& n, long& m,
  int*& out_array, long*& out_degree_list,
  int*& vertex_weights, int*& edge_weights, long& vertex_weights_sum)
{
  ifstream infile;
  string line;
  int format = 0;

  infile.open(filename);
  getline(infile, line); printf("%s\n", line.c_str());
  sscanf(line.c_str(), "%d %li %d", &n, &m, &format);
  m *= 2;
  infile.close();

  bool has_vert_weights = false;
  bool has_edge_weights = false;
  switch(format)
  {
    case  0: break;
    case  1: has_edge_weights = true; break;
    case 10: has_vert_weights = true; break;
    case 11: has_vert_weights = true; has_edge_weights = true; break;
    default:
      fprintf (stderr, "Unknown format specification: '%d'\n", format);
      abort();
  }

  read_adj(filename, n, m, out_array, out_degree_list,
    has_vert_weights, has_edge_weights,
    vertex_weights, edge_weights, vertex_weights_sum);
}


void read_parts(char* filename, int num_verts, int* parts)
{
  ifstream infile;
  string line;
  infile.open(filename);

  for (int i = 0; i < num_verts; ++i)
  {
    getline(infile, line);
    parts[i] = atoi(line.c_str());
  }

  infile.close();
}

void write_parts(char* filename, int num_verts, int* parts)
{
  ofstream outfile;
  outfile.open(filename);

  for (int i = 0; i < num_verts; ++i)
    outfile << parts[i] << endl;

  outfile.close();
}
