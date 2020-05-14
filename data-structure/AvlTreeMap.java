package hw7;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * Map implemented as an AvlTree.
 *
 * @param <K> Type for keys.
 * @param <V> Type for values.
 */
public class AvlTreeMap<K extends Comparable<K>, V>
    implements OrderedMap<K, V> {

  /*** Do not change variable name of 'root'. ***/
  private Node<K, V> root;
  private int size;

  private Node<K, V> insert(Node<K, V> n, K k, V v)
      throws IllegalArgumentException {
    if (n == null) {
      return new Node<>(k, v);
    }

    int cmp = k.compareTo(n.key);
    if (cmp < 0) {
      n.left = insert(n.left, k, v);
    } else if (cmp > 0) {
      n.right = insert(n.right, k, v);
    } else {
      // already mapped
      throw new IllegalArgumentException("duplicate key " + k);
    }

    // This part will only be reached if the new node is inserted
    // either in the left or the right of n, not n itself.
    // Moreover, this statement will be executed on every node whose
    // subtree has been changed by recursion.
    heightAdjustment(n);

    return balance(n);
  }

  @Override
  public void insert(K k, V v) throws IllegalArgumentException {
    if (k == null) {
      throw new IllegalArgumentException("cannot handle null key");
    }
    root = insert(root, k, v);
    size++;
  }

  private Node<K, V> remove(Node<K, V> subtreeRoot, Node<K, V> toRemove) {
    int cmp = subtreeRoot.key.compareTo(toRemove.key);
    if (cmp == 0) {
      return remove(subtreeRoot);
    } else if (cmp > 0) {
      subtreeRoot.left = remove(subtreeRoot.left, toRemove);
    } else {
      subtreeRoot.right = remove(subtreeRoot.right, toRemove);
    }

    heightAdjustment(subtreeRoot);

    return balance(subtreeRoot);
  }

  private Node<K, V> remove(Node<K, V> node) {
    // Easy if the node has 0 or 1 child.
    if (node.right == null) {
      return node.left;
    } else if (node.left == null) {
      return node.right;
    }

    // If it has two children, find the predecessor (max in left subtree),
    Node<K, V> toReplaceWith = maxChild(node);
    // then copy its data to the given node (value change),
    node.key = toReplaceWith.key;
    node.value = toReplaceWith.value;
    // then remove the predecessor node (structural change).
    node.left = remove(node.left, toReplaceWith);

    heightAdjustment(node);

    return balance(node);
  }

  @Override
  public V remove(K k) throws IllegalArgumentException {
    Node<K, V> result = findForSure(k);
    root = remove(root, result);
    size--;
    return result.value;
  }

  @Override
  public void put(K k, V v) throws IllegalArgumentException {
    Node<K, V> n = findForSure(k);
    n.value = v;
  }

  @Override
  public V get(K k) throws IllegalArgumentException {
    Node<K, V> n = find(k);
    if (n == null) {
      throw new IllegalArgumentException("key not mapped");
    }
    return n.value;
  }

  @Override
  public boolean has(K k) {
    if (k == null) {
      return false;
    }
    return find(k) != null;
  }

  @Override
  public int size() {
    return size;
  }

  private void iteratorHelper(Node<K, V> n, List<K> keys) {
    // In order
    if (n == null) {
      return;
    }
    iteratorHelper(n.left, keys);
    keys.add(n.key);
    iteratorHelper(n.right, keys);
  }

  @Override
  public Iterator<K> iterator() {
    List<K> keys = new ArrayList<K>();
    iteratorHelper(root, keys);
    return keys.iterator();
  }

  /*** Do not change this function's name or modify its code. ***/
  // Breadth first traversal that prints binary tree out level by level.
  // Each existing node is printed as follows: 'node.key:node.value'.
  // Non-existing nodes are printed as 'null'.
  // There is a space between all nodes at the same level.
  // The levels of the binary tree are separated by new lines.
  // Returns empty string if the root is null.
  @Override
  public String toString() {
    StringBuilder s = new StringBuilder();
    Queue<Node<K, V>> q = new LinkedList<>();

    q.add(root);
    boolean onlyNullChildrenAdded = root == null;
    while (!q.isEmpty() && !onlyNullChildrenAdded) {
      onlyNullChildrenAdded = true;

      int levelSize = q.size();
      while (levelSize-- > 0) {
        boolean nonNullChildAdded = handleNextNodeToString(q, s);
        if (nonNullChildAdded) {
          onlyNullChildrenAdded = false;
        }
        s.append(" ");
      }

      s.deleteCharAt(s.length() - 1);
      s.append("\n");
    }

    return s.toString();
  }

  /*** Do not change this function's name or modify its code. ***/
  // Helper function for toString() to build String for a single node
  // and add its children to the queue.
  // Returns true if a non-null child was added to the queue, false otherwise
  private boolean handleNextNodeToString(Queue<Node<K, V>> q, StringBuilder s) {
    Node<K, V> n = q.remove();
    if (n != null) {
      s.append(n.key);
      s.append(":");
      s.append(n.value);

      q.add(n.left);
      q.add(n.right);

      return n.left != null || n.right != null;
    } else {
      s.append("null");

      q.add(null);
      q.add(null);

      return false;
    }
  }

  /*** Do not change the name of the Node class.
   * Feel free to add whatever you want to the Node class (e.g. new fields).
   * Just avoid changing any existing names or deleting any existing variables.
   * ***/
  // Inner node class, each holds a key (which is what we sort the
  // BST by) as well as a value. We don't need a parent pointer as
  // long as we use recursive insert/remove helpers.
  // Do not change the name of this class
  private static class Node<K, V> {
    /***  Do not change variable names in this section. ***/
    Node<K, V> left;
    Node<K, V> right;
    K key;
    V value;

    /*** End of section. ***/

    int height;

    // Constructor to make node creation easier to read.
    Node(K k, V v) {
      // left and right default to null
      key = k;
      value = v;
    }

    Node(K k, V v, int h) {
      // left and right default to null
      key = k;
      value = v;
      height = h;
    }

    // Just for debugging purposes.
    public String toString() {
      return "Node<key: " + key
          + "; value: " + value
          + ">";
    }
  }

  private Node<K, V> find(K k) throws IllegalArgumentException {
    if (k == null) {
      throw new IllegalArgumentException("cannot handle null key");
    }
    Node<K, V> n = root;
    while (n != null) {
      int cmp = k.compareTo(n.key);
      if (cmp < 0) {
        n = n.left;
      } else if (cmp > 0) {
        n = n.right;
      } else {
        return n;
      }
    }
    return null;
  }

  private Node<K, V> findForSure(K k) throws IllegalArgumentException {
    Node<K, V> n = find(k);
    if (n == null) {
      throw new IllegalArgumentException("cannot find key " + k);
    }
    return n;
  }

  private int getHeight(Node<K, V> n) {
    if (n == null) {
      return -1;
    }
    return n.height;
  }

  private int getBalanceFactor(Node<K, V> n) {
    return getHeight(n.left) - getHeight(n.right);
  }

  private Node<K, V> balance(Node<K, V> n) {
    if (getBalanceFactor(n) == 2) {
      if (getBalanceFactor(n.left) == -1) {
        n.left = leftRotate(n.left);
      }
      return rightRotate(n);
    } else if (getBalanceFactor(n) == -2) {
      if (getBalanceFactor(n.right) == 1) {
        n.right = rightRotate(n.right);
      }
      return leftRotate(n);
    }

    return n;
  }

  private Node<K, V> leftRotate(Node<K, V> n) {
    // After rotation, n is not root any more
    Node<K, V> ro = n.right;
    Node<K, V> roLR = ro.left;

    ro.left = n;
    n.right = roLR;

    heightAdjustment(n);
    heightAdjustment(ro);

    return ro;
  }

  private Node<K, V> rightRotate(Node<K, V> n) {
    // After rotation, n is not root any more
    Node<K, V> ro = n.left;
    Node<K, V> roRL = ro.right;

    ro.right = n;
    n.left = roRL;

    heightAdjustment(n);
    heightAdjustment(ro);

    return ro;
  }

  private void heightAdjustment(Node<K, V> n) {
    n.height = 1 + Math.max(getHeight(n.left), getHeight(n.right));
  }

  private Node<K, V> maxChild(Node<K, V> n) {
    Node<K, V> curr = n.left; // max in left subtree
    while (curr.right != null) {
      curr = curr.right;
    }
    return curr;
  }
}
