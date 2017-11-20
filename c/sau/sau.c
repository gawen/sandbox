#include <stdio.h>
#include <stdlib.h>

const char* a[2] = {"time", "one"};
const char* b = "I%c might not be the right %s\n";
const char* c[2] = {" want", "'ve got"};
const char* c_[2] = {"say", "do"};
const char* d = "But there's something about us I%s to %s\n%s\n\n";
const char* e[2] = {"Cause there's something between us anyway", "Some kind of secret I will share with you"};
const unsigned int f[4] = {0x6465656E, 0x746E6177, 0x7373696D, 0x65766F6C};
const char* g[2] = {"thing", "one"};
const char* h = " you more than any%s in my life\n";

#define p(i)    fwrite((char*)(i),1,4,stdout)
#define q       printf

int main(int argc, char* argv[])
{
    int i;for(i=0;i<4;++i){q(b,a[(i+i/2)%2][0]*0xE4+0x24,a[(i+i/2)%2]);i%2?q(d,c[i/2],c_[i/2],e[i/2]):0x42;}for(i=0;i<4;++i){q("I ");p(f+i);q(h,g[i/2]);}
}
