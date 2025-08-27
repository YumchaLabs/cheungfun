using UnityEngine;
using System.Collections;

namespace Game.Controllers
{
    /// <summary>
    /// 玩家控制器，处理玩家输入和移动逻辑
    /// </summary>
    [RequireComponent(typeof(Rigidbody2D))]
    [RequireComponent(typeof(Collider2D))]
    public class PlayerController : MonoBehaviour
    {
        [Header("Movement Settings")]
        public float moveSpeed = 5f;
        public float jumpForce = 10f;
        public float maxSpeed = 8f;

        [Header("Ground Check")]
        public Transform groundCheck;
        public float groundCheckRadius = 0.2f;
        public LayerMask groundLayerMask = 1;

        // 组件引用
        private Rigidbody2D rb2d;
        private Collider2D col2d;
        private bool isGrounded;
        private bool facingRight = true;
        private float horizontalInput;

        private void Awake()
        {
            // 获取组件引用
            rb2d = GetComponent<Rigidbody2D>();
            col2d = GetComponent<Collider2D>();
        }

        private void Start()
        {
            // 初始化物理设置
            rb2d.freezeRotation = true;
            rb2d.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
        }

        private void Update()
        {
            HandleInput();
            CheckGrounded();
        }

        private void FixedUpdate()
        {
            HandleMovement();
        }

        /// <summary>
        /// 处理玩家输入
        /// </summary>
        private void HandleInput()
        {
            horizontalInput = Input.GetAxisRaw("Horizontal");

            // 跳跃输入
            if (Input.GetButtonDown("Jump") && isGrounded)
            {
                Jump();
            }
        }

        /// <summary>
        /// 处理玩家移动
        /// </summary>
        private void HandleMovement()
        {
            // 水平移动
            Vector2 velocity = rb2d.velocity;
            velocity.x = horizontalInput * moveSpeed;

            // 限制最大速度
            velocity.x = Mathf.Clamp(velocity.x, -maxSpeed, maxSpeed);
            rb2d.velocity = velocity;

            // 处理角色翻转
            if (horizontalInput > 0 && !facingRight)
            {
                Flip();
            }
            else if (horizontalInput < 0 && facingRight)
            {
                Flip();
            }
        }

        /// <summary>
        /// 跳跃逻辑
        /// </summary>
        private void Jump()
        {
            rb2d.velocity = new Vector2(rb2d.velocity.x, jumpForce);
        }

        /// <summary>
        /// 检查是否在地面上
        /// </summary>
        private void CheckGrounded()
        {
            isGrounded = Physics2D.OverlapCircle(groundCheck.position, groundCheckRadius, groundLayerMask);
        }

        /// <summary>
        /// 翻转角色
        /// </summary>
        private void Flip()
        {
            facingRight = !facingRight;
            Vector3 scale = transform.localScale;
            scale.x *= -1;
            transform.localScale = scale;
        }

        /// <summary>
        /// 重置玩家状态
        /// </summary>
        public void ResetPlayer()
        {
            rb2d.velocity = Vector2.zero;
            transform.position = Vector3.zero;
            facingRight = true;
            transform.localScale = new Vector3(1, 1, 1);
        }

        /// <summary>
        /// 碰撞检测
        /// </summary>
        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.CompareTag("Collectible"))
            {
                CollectItem(other.gameObject);
            }
            else if (other.CompareTag("Enemy"))
            {
                TakeDamage();
            }
        }

        /// <summary>
        /// 收集物品
        /// </summary>
        private void CollectItem(GameObject item)
        {
            Destroy(item);
        }

        /// <summary>
        /// 受到伤害
        /// </summary>
        private void TakeDamage()
        {
            // 击退效果
            StartCoroutine(KnockbackEffect());
        }

        /// <summary>
        /// 击退效果协程
        /// </summary>
        private IEnumerator KnockbackEffect()
        {
            float knockbackForce = 5f;
            rb2d.velocity = new Vector2(-horizontalInput * knockbackForce, jumpForce * 0.5f);

            yield return new WaitForSeconds(0.2f);

            rb2d.velocity = new Vector2(rb2d.velocity.x * 0.5f, rb2d.velocity.y);
        }
    }
}
