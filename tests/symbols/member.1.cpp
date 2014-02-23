
struct point_list {
  int x, y;
  point_list* next;
};

point_list pt;
decltype(pt.x) x;
decltype(pt.y) y;
decltype(pt.next) next;

point_list* pt2;
decltype(pt2->x) x2;
decltype(pt2->y) y2;
decltype(pt2->next) next2;

