<?php
require_once('db.php');

$posts = [];
$error = '';

if (isset($_COOKIE['User'])) {
    $link = mysqli_connect($servername, $username, $password, $dbName);

    if (!$link) {
        $error = 'Ошибка подключения: ' . mysqli_connect_error();
    } else {
        mysqli_set_charset($link, 'utf8mb4');

        $usernameCookie = mysqli_real_escape_string($link, $_COOKIE['User']);
        $sql = "SELECT * FROM posts WHERE username='$usernameCookie' ORDER BY id DESC";
        $res = mysqli_query($link, $sql);

        if ($res) {
            while ($post = mysqli_fetch_array($res)) {
                $posts[] = $post;
            }
        } else {
            $error = 'Не удалось получить посты';
        }

        mysqli_close($link);
    }
}
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Избосарова Ж.О. - главная</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>

<?php if (!isset($_COOKIE['User'])): ?>
<body class="auth-page">

    <div class="pony-card">

        <img src="Gemini_Generated_Image_2gm88z2gm88z2gm8.png" class="pony-avatar" alt="Аватар">

        <h2>PONYGRAM</h2>

        <p class="text-muted mb-4">
            Добро пожаловать в уютное пространство для твоих постов ✨
        </p>

        <div class="d-flex flex-column gap-3">
            <a href="/registration.php" class="btn btn-pink w-100">Регистрация</a>
            <a href="/login.php" class="btn btn-outline-secondary rounded-pill w-100 py-3">Авторизация</a>
        </div>

    </div>

</body>
<?php else: ?>
<body>

    <nav class="navbar navbar-expand-lg navbar-custom fixed-top">

        <div class="container">

            <a class="navbar-brand" href="/index.php">
                PONYGRAM
            </a>

            <div class="d-flex align-items-center gap-3">
                <span class="text-muted">
                    <?php echo htmlspecialchars($_COOKIE['User'], ENT_QUOTES, 'UTF-8'); ?>
                </span>

                <a href="/profile.php" class="btn btn-pink px-4">
                    Профиль
                </a>

                <form action="/logout.php" method="POST" class="d-flex">
                    <button class="btn btn-outline-danger rounded-pill px-4" type="submit">Logout</button>
                </form>
            </div>

        </div>

    </nav>

    <div class="feed-container">

        <div class="create-post-area">
            <h4 class="mb-2">Твои посты ✨</h4>
            <p class="text-muted mb-0">
                На этой странице отображаются публикации пользователя
                <?php echo htmlspecialchars($_COOKIE['User'], ENT_QUOTES, 'UTF-8'); ?>.
            </p>
        </div>

        <?php if ($error !== ''): ?>
            <div class="post-card">
                <p class="mb-0 text-danger"><?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?></p>
            </div>
        <?php elseif (count($posts) > 0): ?>
            <?php foreach ($posts as $post): ?>
                <div class="post-card">

                    <?php if (!empty($post['image_path'])): ?>
                        <div class="post-img-container">
                            <img src="<?php echo htmlspecialchars($post['image_path'], ENT_QUOTES, 'UTF-8'); ?>" alt="<?php echo htmlspecialchars($post['title'], ENT_QUOTES, 'UTF-8'); ?>">
                        </div>
                    <?php endif; ?>

                    <h4><?php echo htmlspecialchars($post['title'], ENT_QUOTES, 'UTF-8'); ?></h4>

                    <p class="text-muted">
                        <?php echo nl2br(htmlspecialchars($post['main_text'], ENT_QUOTES, 'UTF-8')); ?>
                    </p>

                    <div class="d-flex justify-content-between align-items-center">
                        <span class="text-muted">Пост #<?php echo (int) $post['id']; ?></span>

                        <a href="/posts.php?id=<?php echo (int) $post['id']; ?>" class="magic-link">
                            Открыть →
                        </a>
                    </div>

                </div>
            <?php endforeach; ?>
        <?php else: ?>
            <div class="post-card">
                <h4>Пока нет постов</h4>
                <p class="text-muted mb-3">
                    У тебя ещё нет публикаций. Добавь первый пост на странице профиля.
                </p>
                <a href="/profile.php" class="btn btn-pink px-4">Перейти в профиль</a>
            </div>
        <?php endif; ?>

    </div>

    <script src="js/scripts.js"></script>

</body>
<?php endif; ?>
</html>
