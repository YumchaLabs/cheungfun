using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace Game.Managers
{
    /// <summary>
    /// 游戏主管理器，负责游戏状态管理和核心逻辑协调
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        [Header("Game Settings")]
        public float gameSpeed = 1.0f;
        public int maxLives = 3;

        [Header("References")]
        public PlayerController playerController;
        public UIManager uiManager;
        public AudioManager audioManager;

        // 游戏状态
        public enum GameState
        {
            Menu,
            Playing,
            Paused,
            GameOver
        }

        private GameState currentState = GameState.Menu;
        private int currentScore = 0;
        private int currentLives;

        // 单例模式
        public static GameManager Instance { get; private set; }

        private void Awake()
        {
            // 确保只有一个 GameManager 实例
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                InitializeGame();
            }
            else
            {
                Destroy(gameObject);
            }
        }

        private void Start()
        {
            StartCoroutine(GameLoop());
        }

        /// <summary>
        /// 初始化游戏系统
        /// </summary>
        private void InitializeGame()
        {
            currentLives = maxLives;
            currentScore = 0;

            // 初始化各个管理器
            if (uiManager != null)
                uiManager.Initialize();

            if (audioManager != null)
                audioManager.Initialize();
        }

        /// <summary>
        /// 游戏主循环协程
        /// </summary>
        private IEnumerator GameLoop()
        {
            while (true)
            {
                switch (currentState)
                {
                    case GameState.Menu:
                        yield return StartCoroutine(MenuState());
                        break;
                    case GameState.Playing:
                        yield return StartCoroutine(PlayingState());
                        break;
                    case GameState.Paused:
                        yield return StartCoroutine(PausedState());
                        break;
                    case GameState.GameOver:
                        yield return StartCoroutine(GameOverState());
                        break;
                }
                yield return null;
            }
        }

        private IEnumerator MenuState()
        {
            Debug.Log("进入菜单状态");
            yield return new WaitUntil(() => currentState != GameState.Menu);
        }

        private IEnumerator PlayingState()
        {
            Debug.Log("进入游戏状态");
            yield return new WaitUntil(() => currentState != GameState.Playing);
        }

        private IEnumerator PausedState()
        {
            Debug.Log("游戏暂停");
            Time.timeScale = 0f;
            yield return new WaitUntil(() => currentState != GameState.Paused);
            Time.timeScale = gameSpeed;
        }

        private IEnumerator GameOverState()
        {
            Debug.Log("游戏结束");
            yield return new WaitForSeconds(2f);
            RestartGame();
        }

        /// <summary>
        /// 开始游戏
        /// </summary>
        public void StartGame()
        {
            currentState = GameState.Playing;
            currentScore = 0;
            currentLives = maxLives;

            if (playerController != null)
                playerController.ResetPlayer();
        }

        /// <summary>
        /// 暂停游戏
        /// </summary>
        public void PauseGame()
        {
            if (currentState == GameState.Playing)
                currentState = GameState.Paused;
        }

        /// <summary>
        /// 恢复游戏
        /// </summary>
        public void ResumeGame()
        {
            if (currentState == GameState.Paused)
                currentState = GameState.Playing;
        }

        /// <summary>
        /// 重新开始游戏
        /// </summary>
        public void RestartGame()
        {
            currentState = GameState.Menu;
            InitializeGame();
        }

        /// <summary>
        /// 增加分数
        /// </summary>
        public void AddScore(int points)
        {
            currentScore += points;
            if (uiManager != null)
                uiManager.UpdateScore(currentScore);
        }

        /// <summary>
        /// 减少生命值
        /// </summary>
        public void LoseLife()
        {
            currentLives--;
            if (uiManager != null)
                uiManager.UpdateLives(currentLives);

            if (currentLives <= 0)
            {
                currentState = GameState.GameOver;
            }
        }

        // 属性访问器
        public GameState CurrentState => currentState;
        public int CurrentScore => currentScore;
        public int CurrentLives => currentLives;
    }
}
